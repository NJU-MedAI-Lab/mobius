import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.Mobius import MobiusForCausalLM
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, ADD_OTHERS_TOKENS


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--version", default="/workspace/EviMed/huggingface/llava-v1.5-7b"
    )
    parser.add_argument("--pretrain_sam", type=str, default=None)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--vision_pretrained",
                        default="/workspace/EviMed/huggingface/sam-med2d_b.pth", type=str)
    parser.add_argument("--out_dim", default=768, type=int)
    parser.add_argument("--sam_img_size", default=1024,
                        type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument(
        "--vision_tower", default='/workspace/EviMed/huggingface/clip-vit-large-patch14-336', type=str
    )
    parser.add_argument('--imis_sam_checkpoint', type=str,
                        default="/workspace/EviMed/huggingface/IMISNet-B.pth")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules",
                        default="gate_proj,up_proj,down_proj,q_proj,v_proj", type=str)
    parser.add_argument(
        "--sft_modules", default="lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--train_mask_decoder",
                        action="store_true", default=False)
    parser.add_argument("--use_mm_start_end",
                        action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument(
        "--weight", default="/workspace/EviMed/Mobius_runs/mobius-stage4-loop-alllong-bs48-3e-5-5e-5-freezefcs-gradaccu5/ckpt_model_best/global_step128/mp_rank_00_model_states.pt", type=str)
    parser.add_argument(
        "--save_path", default="/workspace/EviMed/checkpoints/mobius-stage4-loop-alllong-bs48-3e-5-5e-5-freezefcs-gradaccu5-128/", type=str)
    parser.add_argument("--region_fea_adapter",
                        action="store_true", default=False)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    for i in range(1, 257):
        ADD_OTHERS_TOKENS.append("<gen_" + str(i) + ">")
    for token_name in ADD_OTHERS_TOKENS:
        tokenizer.add_tokens(token_name, special_tokens=True)
    args.seg_token_idx = tokenizer(
        "<SEG>", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = vars(args)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = MobiusForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    # model.get_model().initialize_imis_modules(model.get_model().config, vars(args))

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                # "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for n, p in model.named_parameters():
            p.requires_grad = False

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    sft_modules = args.sft_modules.split(",")
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in sft_modules
            ]
        ):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    model.resize_token_embeddings(len(tokenizer))

    loaded_state_dict = torch.load(args.weight, map_location="cpu")

    if 'module' in loaded_state_dict:
        print("Detected DeepSpeed checkpoint. Using 'module' key.")
        state_dict = loaded_state_dict['module']
    else:
        state_dict = loaded_state_dict

    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")

    model_dict = model.state_dict()

    filtered_state_dict = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_dict:
            if isinstance(v, torch.Tensor) and v.shape == model_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(
                    f"⚠️ 跳过参数: {k}, 维度不匹配 {v.shape} vs {model_dict[k].shape}")
                skipped_keys.append(k)
        else:
            print(f"⚠️ 未识别的参数: {k}")
            skipped_keys.append(k)

    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict, strict=False)

    print(
        f"✅ 成功加载 {len(filtered_state_dict)} 个匹配的参数，跳过 {len(skipped_keys)} 个参数。")
    if skipped_keys:
        print("跳过的参数包括：")
        for k in skipped_keys[:20]:
            print("  ", k)

    if hasattr(model, "merge_and_unload"):
        print("Merging LoRA weights into the base model...")
        model = model.merge_and_unload()
    else:
        print("⚠️ 未检测到LoRA模块，跳过merge_and_unload。")
    state_dict = {}
    for k, v in model.state_dict().items():
        # if "vision_tower" not in k:
        state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
