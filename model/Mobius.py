import glob
import os
from typing import Any, Dict, List, Tuple
from typing import Dict, Optional, Sequence, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)
from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, REGION_TOKEN_INDEX

from .mobius.model.language_model.mobius_llama import (LlavaLlamaForCausalLM,
                                                       LlavaLlamaModel)

from .segment_anything_med2d import build_sam_vit_b
from .IMIS.build_imis import build_imis_model
import re
import warnings
warnings.filterwarnings("ignore")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * \
            w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / \
            (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_loss


class MaskMSE(nn.Module):
    def __init__(self, ):
        super(MaskMSE, self).__init__()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        mse = torch.mean((iou - pred_iou) ** 2)
        return mse


class FocalDice_MSELoss(nn.Module):
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDice_MSELoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_mse = MaskMSE()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss = self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_mse(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss


class MobiusMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MobiusMetaModel, self).__init__(config)

        self.config = config
        self.initialize_imis_modules(config, kwargs)

    def initialize_imis_modules(self, config, kwargs):
        # IMIS Model
        self.visual_model = build_imis_model(kwargs)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        if config.train_prompt_encoder:
            self.visual_model.prompt_encoder.train()
            for param in self.visual_model.prompt_encoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        # Projection layer: from IMIS to VLM
        # config.hidden_size  # IMIS mask_tokens_out.shape[1]=768
        this_in_dim = 768
        this_out_dim = 4096  # config.hidden_size
        imis_to_vlm_proj_layers = [
            nn.Linear(this_in_dim, this_in_dim),
            nn.GELU(),
            nn.Linear(this_in_dim, this_out_dim),
            nn.Dropout(0.0),
        ]
        self.imis_to_vlm_proj_fcs = nn.Sequential(*imis_to_vlm_proj_layers)
        self.imis_to_vlm_proj_fcs.train()
        for param in self.imis_to_vlm_proj_fcs.parameters():
            param.requires_grad = True

        # Load pretrained weights for IMIS modules and projection layers
        if 'pretrain_sam' in kwargs and kwargs['pretrain_sam'] is not None:
            imis_ckpt = {}
            bin_files = glob.glob(os.path.join(
                kwargs['pretrain_sam'], '*.bin'))
            for file in bin_files:
                part_ckpt = torch.load(file, map_location='cpu')
                imis_ckpt.update(part_ckpt)
            print("Loading IMIS weights from", kwargs['pretrain_sam'])
            # print("Keys in IMIS checkpoint:", imis_ckpt.keys())

            target_dict = {}
            for k, v in imis_ckpt.items():
                if any([target in k for target in ['mask_decoder']]):
                    target_dict[k.replace(
                        'model.visual_model.mask_decoder.', '')] = v
            load_res = self.visual_model.mask_decoder.load_state_dict(
                target_dict, strict=False)
            print('the mask decoder weights load results', load_res)

            target_dict = {}
            for k, v in imis_ckpt.items():
                if any([target in k for target in ['text_hidden_fcs']]):
                    target_dict[k.replace(
                        'model.text_hidden_fcs.', '')] = v
            load_res = self.text_hidden_fcs.load_state_dict(
                target_dict, strict=False)
            print('the text hidden fcs weights load results', load_res)

            target_dict = {}
            for k, v in imis_ckpt.items():
                if any([target in k for target in ['imis_to_vlm_proj_fcs']]):
                    target_dict[k.replace(
                        'model.imis_to_vlm_proj_fcs.', '')] = v
            load_res = self.imis_to_vlm_proj_fcs.load_state_dict(
                target_dict, strict=False)
            print('the imis to vlm proj fcs weights load results', load_res)
        else:
            print("No pretrained IMIS weights provided.")


class MobiusModel(MobiusMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MobiusModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        # self.config.pretrain_mm_mlp_adapter = kwargs.get(
        #     'pretrain_mm_mlp_adapter', None)
        # self.config.pretrain_mm_mlp_adapter = "/workspace/EviMed/huggingface/llava-llama-2-7b-chat-clip336-pretrain-medplib-stage1/mm_projector.bin"
        self.config.mm_use_im_patch_token = False


class MobiusForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        tokenizer=None,
        **kwargs,
    ):
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        if not hasattr(config, "vision_tower"):
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14")
        else:
            config.mm_vision_tower = config.vision_tower
        self.vlm_ce_loss_weight = kwargs.pop("vlm_ce_loss_weight", None)
        self.sam_ce_loss_weight = kwargs.pop("sam_ce_loss_weight", None)
        self.vlm_mask_loss_weight = kwargs.pop("vlm_mask_loss_weight", None)
        self.sam_mask_loss_weight = kwargs.pop("sam_mask_loss_weight", None)
        config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
        config.train_prompt_encoder = kwargs.get("train_prompt_encoder", True)
        config.out_dim = kwargs.get("out_dim", 768)

        self.seg_token_idx = kwargs.pop("seg_token_idx")

        config.region_fea_adapter = kwargs.get("region_fea_adapter", False)
        config.region_geo_sampler = kwargs.get("region_geo_sampler", False)
        config.max_sample_point = kwargs.get("max_sample_point", 512)
        config.sampler_pooler_mode = kwargs.get("sampler_pooler_mode", 'max')

        super().__init__(config)
        # print('LISAForCausalLM config', config)

        self.model = MobiusModel(config, **kwargs)
        self.tokenizer = tokenizer

        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.set_loss_fn()
        print("IMIS and projection layers are added.")

    def set_loss_fn(self):
        self.seg_loss = FocalDice_MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings  # [B, C, H, W]

    def IMIS_forward(self, image_embeddings, prompt, labels, return_logits=True):
        # Prompts
        # if prompt.get("point_coords", None) is None:
        #     points = None
        # else:
        #     points = (prompt["point_coords"], prompt["point_labels"])

        sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text=prompt.get("text_inputs", None),
        )

        outputs = self.model.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
            text_prompt_embeddings=prompt.get("text_inputs", None),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_masks = outputs['low_res_masks']  # [B, 1, 256, 256]
        mask_tokens_out = outputs['mask_tokens_out']
        iou_pred = outputs['iou_pred']
        semantic_pred = outputs['semantic_pred']

        masks = F.interpolate(low_res_masks, size=self.model.visual_model.image_size, mode='bilinear',
                              align_corners=False)

        mask_loss = self.seg_loss(masks.float(), labels.float(), iou_pred)

        if return_logits:
            mask_logits = F.interpolate(low_res_masks, size=self.model.visual_model.image_size, mode='bilinear',
                                        align_corners=False)
        else:
            mask_logits = None

        outputs = {
            'low_res_masks': low_res_masks,
            'mask_tokens_out': mask_tokens_out,
            'masks': masks,  # soft mask (for diff. path)
            'mask_logits': mask_logits,  # logits for sampling
            'mask_loss': mask_loss,
            'iou_pred': iou_pred,
            'semantic_pred': semantic_pred,
        }
        return outputs

    def expand_batch_for_multiseg(
        self,
        image_embeddings,
        proj_embeddings,
        input_ids,
        seg_token_idx,
        prompts,
        imis_labels,
        masks_list,
        resize_list,
        label_list
    ):
        B = image_embeddings.shape[0]

        num_qs = [(input_ids[b] == seg_token_idx).sum().item()
                  for b in range(B)]
        total_q = sum(num_qs)

        expand_image_embeddings = []
        expand_point_coords = []
        expand_point_labels = []
        expand_bboxes = []
        expand_mask_inputs = []
        expand_gt_masks = []
        expand_masks_list = []
        expand_resize_list = []
        expand_label_list = []

        seg_idx = 0
        for b in range(B):
            n = num_qs[b]
            if n == 0:
                continue

            # expand image_embeddings
            expand_image_embeddings.extend([image_embeddings[b]]*n)

            # expand prompts
            if prompts.get("point_coords", None) is not None:
                expand_point_coords.extend(
                    [(prompts["point_coords"][b].unsqueeze(0))]*n)

            if prompts.get("point_labels", None) is not None:
                expand_point_labels.extend(
                    [(prompts["point_labels"][b].unsqueeze(0))]*n)

            if prompts.get("bboxes", None) is not None:
                expand_bboxes.extend([(prompts["bboxes"][b].unsqueeze(0))]*n)

            if prompts.get("mask_inputs", None) is not None:
                expand_mask_inputs.extend(
                    [(prompts["mask_inputs"][b].unsqueeze(0))]*n)

            expand_gt_masks.extend([imis_labels[b].float()]*n)
            expand_masks_list.extend([masks_list[b]]*n)
            expand_resize_list.extend([resize_list[b]]*n)
            expand_label_list.extend([label_list[b]]*n)

            seg_idx += n

        expand_image_embeddings = torch.stack(expand_image_embeddings, dim=0)
        expand_gt_masks = torch.cat(expand_gt_masks, dim=0).unsqueeze(1)
        expand_point_coords = torch.cat(
            expand_point_coords, dim=0) if expand_point_coords is not None and len(expand_point_coords) > 0 else None
        expand_point_labels = torch.cat(
            expand_point_labels, dim=0) if expand_point_labels is not None and len(expand_point_labels) > 0 else None
        expand_bboxes = torch.cat(
            expand_bboxes, dim=0) if expand_bboxes is not None and len(expand_bboxes) > 0 else None
        expand_mask_inputs = torch.cat(
            expand_mask_inputs, dim=0) if expand_mask_inputs is not None and len(expand_mask_inputs) > 0 else None
        expand_text_embs = torch.stack(
            proj_embeddings[:total_q], dim=0).squeeze(1)

        expand_prompts = {
            'point_coords': expand_point_coords,
            'point_labels': expand_point_labels,
            'bboxes': expand_bboxes,
            'mask_inputs': expand_mask_inputs,
            'text_inputs': expand_text_embs
        }

        return {
            "image_embeddings": expand_image_embeddings,
            "gt_masks": expand_gt_masks,
            "prompts": expand_prompts,
            "masks_list": expand_masks_list,
            "resize_list": expand_resize_list,
            "label_list": expand_label_list,
        }

    def process_text_prompt(self, classes):
        bs_text_prompt = self.text_tokenizer(classes)
        return {'text_inputs': bs_text_prompt.to(self.device)}

    def text_tokenizer(self, text, tamplate='A segmentation area of a {}.'):
        # 得到规范化文本列表
        norm_text = []
        for t in text:
            t = self.model.visual_model.categories_map[t][0]
            t = t.lower().replace('_', ' ').replace("-", " ")
            t = re.sub(r'\s+', ' ', t)
            norm_text.append(t)
        text_list = [tamplate.format(t)
                     for t in norm_text]  # 按照template得到标准prompt
        tokens = self.tokenizer(text_list, padding=True,
                                return_tensors="pt")  # 使用之前定义的tokenizer
        for key in tokens.keys():
            tokens[key] = tokens[key].to(self.device)
        text_outputs = self.model.visual_model.text_model(
            **tokens)  # sam text model
        text_embedding = text_outputs.pooler_output  # pooled output
        text_embedding = self.model.visual_model.text_out_dim(text_embedding)
        return text_embedding  # text_embedding.shape[1]=768

    def forward(self, mode="vlm_sam", **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)

        vlm_only = kwargs.get('vlm_only', False)
        sam_only = kwargs.get('sam_only', False)
        proj_only = kwargs.get('proj_only', False)

        # 初始化损失
        if not proj_only:
            total_loss = torch.tensor(
                0.0, device=kwargs['images_clip'].device)
        else:
            total_loss = torch.tensor(
                0.0, device=kwargs['images_clip'].device)

        if vlm_only:
            return self.model_forward_vlm(**kwargs)

        if sam_only:
            return self.model_forward_vlm_sam(**kwargs)

        if mode == "sam_vlm":
            loss_dict = self.model_forward_sam_vlm(**kwargs)
        elif mode == "vlm_sam":
            loss_dict = self.model_forward_vlm_sam(**kwargs)
        else:
            raise ValueError("mode must be sam_vlm or vlm_sam")

        total_loss = loss_dict["mask_loss"] + loss_dict["ce_loss"]
        loss_dict["loss"] = total_loss

        return loss_dict

    def model_forward_vlm(
        self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,  # question+answer token for training
        attention_masks: torch.LongTensor,
        labels: torch.LongTensor,
        inference: bool = False,
        region_masks: List[torch.FloatTensor] = None,
        valid_region_masks_bool: Optional[List[torch.BoolTensor]] = [],
        **kwargs,
    ):
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            region_masks=region_masks,
            valid_region_masks_bool=valid_region_masks_bool,
        )  # llama forward
        ce_loss = output.loss
        return {
            "loss": ce_loss,
            "ce_loss": ce_loss,
            "mask_loss": torch.tensor(0.0).to(images_clip.device),
        }

    def model_forward_vlm_sam(
            self,
            images_clip: torch.FloatTensor,
            images_imis: torch.FloatTensor,
            input_ids: torch.LongTensor,  # question+answer token for training
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            masks_list: List[torch.FloatTensor],
            label_list: List[torch.Tensor],
            resize_list: List[tuple],
            imis_gt_masks: torch.LongTensor,  # gt imis masks, 一个回答中可能有多个masks
            valid_mask_bool,
            sam_only: bool = False,
            inference: bool = False,
            **kwargs,
    ):
        """
        VLM → SAM 双向监督：
            - SAM 通过 mask_loss 训练（监督信号来自 masks_list）
            - VLM 通过 ce_loss + 感知一致性 reward 训练
        """
        prompts = {
            'point_coords': None, 'point_labels': None, 'bboxes': None, 'mask_inputs': None, 'text_inputs': None
        }

        B = input_ids.shape[0]  # b*q_num

        # ------------------ 1. VLM Forward: Get Hidden States ------------------
        # vlm->imis阶段不需要region masks的输入
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            region_masks=None,
            valid_region_masks_bool=None,
        )  # llama forward
        if not sam_only:
            ce_loss = output.loss * self.vlm_ce_loss_weight
        last_hidden_state = output.hidden_states[-1]  # [B*q_num, N, 4096]

        # ------------------ 2. Extract VLM Answer Embeddings ------------------
        input_ids = input_ids[valid_mask_bool]  # [B_valid, N]
        # [B_valid, N, 4096]
        last_hidden_state = last_hidden_state[valid_mask_bool]
        images_imis = images_imis[valid_mask_bool]

        seg_token_mask = (input_ids[:, 1:] == self.seg_token_idx)
        seg_token_mask = torch.cat([seg_token_mask, torch.zeros(
            (seg_token_mask.shape[0], 1)).bool().to(input_ids.device)], dim=1)
        seg_token_mask = torch.cat([
            torch.zeros((seg_token_mask.shape[0], 575)).bool().to(
                input_ids.device),
            seg_token_mask
        ], dim=1)  # [B_valid, 575 + seq_len]

        if inference:
            last_hidden_state = last_hidden_state.unsqueeze(0)
        pred_embeddings_full = last_hidden_state[seg_token_mask]
        pred_embeddings_list = torch.split(pred_embeddings_full, 1)
        proj_embeddings = [self.model.text_hidden_fcs[0](
            emb) for emb in pred_embeddings_list]

        # ------------------ 3. IMIS Forward: Use VLM's Answer as Text Prompt ------------------
        image_embeddings = self.get_visual_embs(
            images_imis)  # [B_valid, 256, 64, 64]

        expand_data = self.expand_batch_for_multiseg(
            image_embeddings,
            proj_embeddings,
            input_ids,
            self.seg_token_idx,
            prompts,
            imis_gt_masks,
            masks_list,
            resize_list,
            label_list
        )

        imis_output = self.IMIS_forward(
            expand_data["image_embeddings"],
            expand_data["prompts"],
            expand_data["gt_masks"],
            return_logits=True
        )

        pred_masks = []
        for i in range(imis_output['low_res_masks'].shape[0]):
            pred_mask = self.postprocess_masks(
                imis_output['low_res_masks'][i],
                input_size=expand_data['resize_list'][i],
                original_size=expand_data['label_list'][i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        gt_masks = expand_data['masks_list']

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        mask_loss = imis_output["mask_loss"] * self.vlm_mask_loss_weight

        # ------------------ 5. Total Loss ------------------
        if sam_only:
            return {
                "loss": mask_loss,
                "ce_loss": torch.tensor(0.0).to(mask_loss.device),
                "mask_loss": mask_loss,
            }

        total_loss = ce_loss + mask_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
        }

    def model_forward_sam_vlm(
            self,
            images_imis: torch.FloatTensor,
            images_clip: torch.FloatTensor,
            input_ids_seg: torch.LongTensor,
            labels_seg: torch.LongTensor,
            attention_masks_seg: torch.LongTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            imis_gt_masks: torch.LongTensor,
            proj_only: bool = False,
            **kwargs,
    ):
        # prompts = None
        B = images_imis.shape[0]

        # ------------------ 1. VLM: generate prompts for IMIS ------------------
        with torch.no_grad():
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks_seg,
                input_ids=input_ids_seg,
                labels=labels_seg,
                output_hidden_states=True,
                region_masks=None,
                valid_region_masks_bool=None,
            )  # llama forward

            last_hidden_state = output.hidden_states[-1]  # [B*q_num, N, 4096]

            seg_token_mask = (input_ids_seg[:, 1:] == self.seg_token_idx)
            seg_token_mask = torch.cat([seg_token_mask, torch.zeros(
                (seg_token_mask.shape[0], 1)).bool().to(input_ids_seg.device)], dim=1)
            seg_token_mask = torch.cat([
                torch.zeros((seg_token_mask.shape[0], 575)).bool().to(
                    input_ids_seg.device),
                seg_token_mask
            ], dim=1)  # [B*q_num, 575 + seq_len]

            pred_embeddings_full = last_hidden_state[seg_token_mask]
            # 拆分成list，每个list长度为1，即每个list代表一个query的answer embedding
            pred_embeddings_list = torch.split(pred_embeddings_full, 1)
            proj_embeddings = [self.model.text_hidden_fcs[0](
                emb) for emb in pred_embeddings_list]
            proj_embeddings = torch.stack(
                proj_embeddings, dim=0).squeeze(1)

        # ------------------ 1. IMIS: Forward + Get mask embeddings ------------------
        # add short text inputs
        prompts = {
            'point_coords': None, 'point_labels': None, 'bboxes': None, 'mask_inputs': None, 'text_inputs': proj_embeddings
        }

        # 修改：mask_decoder 输出 raw logits（未激活）
        image_embeddings = self.get_visual_embs(
            images_imis)  # [B*q_num, 256, 64, 64]
        sam_outputs = self.IMIS_forward(
            image_embeddings, prompts, imis_gt_masks, return_logits=True
        )
        if proj_only:
            mask_loss = torch.tensor(0.0).to(images_imis.device)
        else:
            mask_loss = sam_outputs["mask_loss"] * \
                self.sam_mask_loss_weight  # 监督 loss

        region_masks = self.model.imis_to_vlm_proj_fcs(
            sam_outputs['mask_tokens_out'])
        valid_region_masks_bool = [[torch.ones(1).bool().to(
            region_masks.device)]]*region_masks.shape[0]

        # ------------------ 2. VLM: Forward ------------------
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            region_masks=region_masks,
            valid_region_masks_bool=valid_region_masks_bool,
            is_embeds=True,
        )
        ce_loss = output.loss * self.sam_ce_loss_weight

        # ------------------ 3. Total Loss ------------------
        total_loss = ce_loss + mask_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
        }

    def evaluate_vlm(
        self,
        images_clip,
        input_ids,
        attention_masks,
        region_masks=[],
        valid_region_masks_bool: Optional[List[torch.BoolTensor]] = [],
        max_new_tokens=1024,
        tokenizer=None,
        **kwargs,
    ):
        indices = (input_ids == 29901).nonzero(as_tuple=True)
        input_ids = input_ids[:, :indices[1][-1]+1]
        attention_masks = attention_masks[:, :indices[1][-1]+1]
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                region_masks=region_masks,
                valid_region_masks_bool=valid_region_masks_bool,
                attention_mask=attention_masks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        return outputs

    def evaluate_sam_vlm(
        self,
        images_clip,
        images_imis,
        input_ids_seg,
        attention_masks_seg,
        input_ids,
        attention_masks,
        **kwargs
    ):
        indices = (input_ids_seg == 29901).nonzero(as_tuple=True)
        input_ids_seg = input_ids_seg[:, :indices[1][-1]+1]
        attention_masks_seg = attention_masks_seg[:, :indices[1][-1]+1]
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids_seg,
                attention_mask=attention_masks_seg,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # B*q_num, N, 256 --> B*q_num, 255+sequence_length, 256
            # last_hidden_state = last_hidden_state[:, :seg_token_mask.shape[1], :]
            pred_embeddings = last_hidden_state[seg_token_mask]

            if pred_embeddings.shape[0] > 1:
                pred_embeddings = pred_embeddings[:1, :]
            elif pred_embeddings.shape[0] == 0:
                pred_embeddings = last_hidden_state[:1, -2:-1, :].squeeze(1)

            image_embeddings = self.get_visual_embs(images_imis)
            pred_masks = []
            sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text=pred_embeddings[0].unsqueeze(0),
            )

            sparse_embeddings = sparse_embeddings.to(
                pred_embeddings[0].dtype)

            outputs = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                text_prompt_embeddings=pred_embeddings[0].unsqueeze(
                    0),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            region_masks = self.model.imis_to_vlm_proj_fcs(
                outputs['mask_tokens_out'])
            valid_region_masks_bool = [torch.ones(
                region_masks.shape[0]).bool().to(region_masks.device)]

            indices = (input_ids == 29901).nonzero(as_tuple=True)
            input_ids = input_ids[:, :indices[1][-1]+1]
            attention_masks = attention_masks[:, :indices[1][-1]+1]
            vlm_outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                region_masks=region_masks,
                valid_region_masks_bool=valid_region_masks_bool,
                is_embeds=True,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                attention_mask=attention_masks,
            )

            vlm_output_ids = vlm_outputs.sequences

            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                vlm_output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
        return outputs

    def evaluate(
        self,
        images_clip,
        images_imis,
        input_ids,
        resize_list,
        attention_masks,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        max_new_tokens=1024,
        **kwargs
    ):
        indices = (input_ids == 29901).nonzero(as_tuple=True)
        input_ids = input_ids[:, :indices[1][-1]+1]
        attention_masks = attention_masks[:, :indices[1][-1]+1]
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                region_masks=None,
                valid_region_masks_bool=None,
                attention_mask=attention_masks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(
                self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            # B*q_num, N, 256 --> B*q_num, 255+sequence_length, 256
            # last_hidden_state = last_hidden_state[:, :seg_token_mask.shape[1], :]
            pred_embeddings = last_hidden_state[seg_token_mask]

            if pred_embeddings.shape[0] > 1:
                pred_embeddings = pred_embeddings[:1, :]
            elif pred_embeddings.shape[0] == 0:
                pred_embeddings = last_hidden_state[:1, -2:-1, :].squeeze(1)

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            image_embeddings = self.get_visual_embs(images_imis)

            multimask_output = False
            pred_masks = []
            gt_masks = masks_list
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text=pred_embeddings[i].unsqueeze(0),
                )

                sparse_embeddings = sparse_embeddings.to(
                    pred_embeddings[i].dtype)
                outputs = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    text_prompt_embeddings=pred_embeddings[i].unsqueeze(
                        0),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.postprocess_masks(
                    outputs["low_res_masks"],
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])

            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()

        return outputs, pred_masks, gt_masks

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]):
        # """
        # Removes padding from a padded tensor and retrieves the original tensor shape.

        # Returns:
        # torch.Tensor: The original image tensor with shape (b, 3, original_height, original_width).
        # """
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(0)

        masks = F.interpolate(masks, original_size,
                              mode="bilinear", align_corners=False)
        return masks
