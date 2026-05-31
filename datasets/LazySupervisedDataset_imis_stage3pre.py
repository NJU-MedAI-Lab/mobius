import ast
import types
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import json
import copy
import os
import re

from scipy import sparse
from torch.utils.data import Dataset
import transformers
import torch
import torch.nn.functional as F
from PIL import Image
# from torchvision import transforms
from monai import data, transforms
import numpy as np
import cv2
import random
import time
import torchvision

from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, REGION_TOKEN_INDEX
from model.mobius import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
import traceback
from tqdm import tqdm

from datasets.data_utils import (
    Resize,
    PermuteTransform,
    LongestSidePadding,
    Normalization,
    get_points_from_mask,
    get_bboxes_from_mask
)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    need_region: bool = True,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, need_region=need_region)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """
    Preprocess multimodal data.

    Args:
        sources (Sequence[str]): A sequence of strings representing the raw multimodal data.
        data_args (DataArguments): A data arguments object containing the necessary parameters for multimodal data processing.

    Returns:
        Dict: The preprocessed multimodal data in the form of a dictionary.

    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in str(sentence['value']):
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def truncate_conversation(conv: str) -> str:
    marker = "ASSISTANT:"
    idx = conv.find(marker)
    if idx != -1:  # 找到标记
        return conv[:idx+10]
    return conv  # 没找到就返回原文


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    need_region: bool = True,
) -> Dict:
    """
    Preprocess the input data for the model.

    Args:
        sources (List[Dict[str, str]]): A list of dictionaries, each representing a conversation turn with keys "from" and "value".
        tokenizer (transformers.PreTrainedTokenizer): A pre-trained tokenizer object from the transformers library.
        has_image (bool, optional): Whether the input data contains images. Defaults to False.

    Returns:
        Dict: A dictionary containing preprocessed data, including:
            - input_ids (torch.Tensor): Tokenized input IDs.
            - labels (torch.Tensor): Labels for masked targets.
            - conversations (List[str]): The original conversations.
            - question (List[str]): Extracted questions from the conversations.
            - gt (List[str]): Extracted ground truth responses from the conversations.

    """
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    question = []
    gt = []
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            # record the question and answer
            if sentence['from'] == 'human':
                question.append(sentence['value'].replace(
                    '<im_start><image><im_end>\n', ''))
            else:
                gt.append(sentence['value'])

            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())  # 按照模版合并好的conversation

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets, mask the question part of the conversation
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print('tokenization mismatch, the question is', question)
                print('tokenization mismatch, the conversations is', conversations)
    return dict(
        input_ids=input_ids,  # 模型输入 token 序列
        labels=targets,  # mask问题后的目标 token 序列
        conversations=conversations,  # 拼接后的原始prompt
        question=question,  # 提取的问题
        gt=gt,  # 提取的真实回答
    )


def process_mask(mask):
    mask_tensor = torch.tensor(mask, dtype=torch.float)

    return mask_tensor


def generate_sub_connected_component(component, min_area, max_area, min_thresh=1000):
    # Calculate the area of the current connected component
    component_area = np.sum(component == 1)
    # Randomly select the ratio of the sub-connected component's area
    target_area = 0
    if component_area < min_thresh:
        # print('This component_area is too small', component_area)
        return component

    while target_area // min_thresh < 1:
        target_ratio = random.uniform(min_area, max_area)
        # Calculate the target area of the sub-connected component
        target_area = int(component_area * target_ratio)

    # Generate a new sub-connected component within the current connected component
    sub_component = np.zeros_like(component)

    # Randomly select a starting point
    row, col = np.where(component == 1)
    start_point = random.choice(list(zip(row, col)))

    stack = [start_point]
    while len(stack) > 0:
        current_point = stack.pop()
        sub_component[current_point] = 1

        # Check if the area of the sub-connected component reaches the target area
        if np.sum(sub_component == 1) >= target_area:
            break

        # Randomly select a neighbor around the current point as the next point
        neighbors = [(current_point[0] + dy, current_point[1] + dx)
                     for dy in [-1, 0, 1] for dx in [-1, 0, 1]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if 0 <= neighbor[0] < component.shape[0] and 0 <= neighbor[1] < component.shape[1] and component[neighbor] == 1 and sub_component[neighbor] == 0:
                stack.append(neighbor)

    return sub_component


def generate_mask_with_sub_component(masks, min_area=0.4, max_area=1.0, min_thresh=1000):
    sub_components = []
    is_valid = False
    for mask in masks:
        mask = np.array(mask)
        if np.sum(mask) > 0:
            # Get the connected components
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

            # Randomly select a connected component
            label_values = np.unique(labels)[1:]  # Remove background label 0

            max_area = 0
            max_area_label = 0
            for label_value in label_values:
                area = np.sum(labels == label_value)
                if area > max_area:
                    max_area = area
                    max_area_label = label_value
                    # print(max_area, 'max_area')
                is_valid = True
                # selected_label = random.choice(label_values)
                selected_label = max_area_label

                # Get the current connected component
                current_component = np.where(labels == selected_label, 1, 0)

                # Generate a sub-connected component
                sub_component = generate_sub_connected_component(
                    current_component, min_area=min_area, max_area=max_area, min_thresh=min_thresh)
        else:
            is_valid = False
            sub_component = np.ones((336, 336))

        sub_components.append(sub_component)
    return sub_components, is_valid


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    # 按<image>拆分文本并编码
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    # 在chunk之间插入<image> token
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # 如果第一个chunk以BOS token 开头，就保留下来，设offset=1表示跳过第一个token
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # 把<image> 占位符替换成image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    i = 0
    element1 = tokenizer("<region>", add_special_tokens=False).input_ids[0]
    element2 = tokenizer("</region>", add_special_tokens=False).input_ids[0]
    # 找到形如<region> </region>的空区域，在中间插入一个region_token_index
    while i < len(input_ids) - 1:
        if input_ids[i] == element1 and input_ids[i + 1] == element2:
            input_ids.insert(i + 1, REGION_TOKEN_INDEX)
            i += 1
        i += 1

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


class LazySupervisedDataset_stage3pre(Dataset):
    """Dataset for supervised fine-tuning."""

    # for sam
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # for clip
    clip_pixel_mean = (torch.Tensor(
        [0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)
    clip_pixel_std = (torch.Tensor(
        [0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)

    ignore_label = 255

    def __init__(self, data_path: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_size=1024,
                 proj_only: bool = False,
                 mode='train'):
        super(LazySupervisedDataset_stage3pre, self).__init__()
        self.proj_only = proj_only
        list_data_dict = json.load(open(data_path[0], "r"))
        self.list_data_dict = []
        if not proj_only:
            list_data_dict_vlm = json.load(open(data_path[1], "r"))
            self.list_data_dict_vlm = []
            assert len(list_data_dict) == len(
                list_data_dict_vlm), "The length of two dataset should be the same"
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.sam_img_size = image_size
        self.clip_img_size = 336

        self.transform_imis = transforms.Compose(
            [
                Resize(keys=["image", "mask"],
                       target_size=(self.sam_img_size, self.sam_img_size)),
                PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                transforms.ToTensord(keys=["image", "mask"]),
                Normalization(keys=["image"]),
                transforms.RandScaleIntensityd(
                    keys="image", factors=0.2, prob=0.2),
                transforms.RandShiftIntensityd(
                    keys="image", offsets=0.2, prob=0.2),
            ]
        )
        self.transform_clip = transforms.Compose(
            [
                Resize(keys=["image", "mask"],
                       target_size=(self.clip_img_size, self.clip_img_size)),
                PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                transforms.ToTensord(keys=["image", "mask"]),
            ]
        )

        if mode == 'train':
            split = 'training'
        elif mode == 'test':
            split = 'test'
        else:
            split = 'validation'

        for i, item in enumerate(tqdm(list_data_dict[split], desc="Loading and validating long data")):
            # --- 图片路径处理保持不变 ---
            if 'image' in item:
                image_file = item['image']
                if not os.path.isfile(image_file):
                    # print("no such image file:" + image_path)
                    continue  # 图片不存在，跳过
                else:
                    item['image'] = image_file
            else:
                print("No image field in item")
                continue  # 没有图片字段，跳过？

            # --- 确保对话内容是字符串 ---
            for idx, _ in enumerate(item['conversations']):
                if not isinstance(item['conversations'][idx]['value'], str):
                    item['conversations'][idx]['value'] = str(
                        item['conversations'][idx]['value'])

            for idx, _ in enumerate(item['conversations']):
                if item['conversations'][idx]['from'] == 'gpt' and '<SEG>' not in item['conversations'][idx]['value']:
                    print('No <SEG> in gpt answer, skip this sample')
                    continue

            self.list_data_dict.append(item)  # 保存包含路径信息的 item

        print(f"{self.__len__()} for imis->vlm samples loaded after validation.")

        if not proj_only:
            for i, item in enumerate(tqdm(list_data_dict_vlm[split], desc="Loading and validating long data")):
                # --- 图片路径处理保持不变 ---
                if 'image' in item:
                    image_file = item['image']
                    if not os.path.isfile(image_file):
                        # print("no such image file:" + image_path)
                        continue  # 图片不存在，跳过
                    else:
                        item['image'] = image_file
                else:
                    print("No image field in item")
                    continue  # 没有图片字段，跳过？

                # --- 确保对话内容是字符串 ---
                for idx, _ in enumerate(item['conversations']):
                    if not isinstance(item['conversations'][idx]['value'], str):
                        item['conversations'][idx]['value'] = str(
                            item['conversations'][idx]['value'])

                for idx, _ in enumerate(item['conversations']):
                    if item['conversations'][idx]['from'] == 'gpt' and '<SEG>' not in item['conversations'][idx]['value']:
                        print('No <SEG> in gpt answer, skip this sample')
                        continue

                self.list_data_dict_vlm.append(item)  # 保存包含路径信息的 item

            print(
                f"{len(self.list_data_dict_vlm)} for vlm->imis samples loaded after validation.")

        self.label_mapping = json.load(
            open(os.path.join(self.data_args.image_folder, "label_mapping.json"), 'r'))

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def pad_tensor_channelwise(self, x, pad_h, pad_w, pad_values, is_mask=False):
        """
        Pad a 3-channel image tensor with different padding values for each channel,
        considering total padding length and odd padding size.

        Parameters:
        x (torch.Tensor): Input image tensor of shape (3, h, w).
        pad_h (int): Total padding size for the height.
        pad_w (int): Total padding size for the width.
        pad_values (tuple): A tuple of three elements specifying the padding value for each channel.

        Returns:
        torch.Tensor: Padded image tensor.
        """

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if is_mask:
            assert len(
                pad_values) == 1, "pad_values must have 1 elements, one for each channel."
            padded_tensor = torch.empty(
                (x.shape[0] + pad_h, x.shape[1] + pad_w), dtype=x.dtype)
            padded_tensor[:, :] = pad_values[0]
            padded_tensor[pad_top:pad_top+x.shape[0],
                          pad_left:pad_left+x.shape[1]] = x
        else:
            assert len(
                pad_values) == 3, "pad_values must have three elements, one for each channel."
            padded_tensor = torch.empty(
                (3, x.shape[1] + pad_h, x.shape[2] + pad_w), dtype=x.dtype)
            for i in range(3):
                padded_tensor[i, :, :] = pad_values[i]
            padded_tensor[:, pad_top:pad_top+x.shape[1],
                          pad_left:pad_left+x.shape[2]] = x

        return padded_tensor

    def preprocess(self, x: torch.Tensor, image_size: int, normalize: bool = True, is_mask: bool = False) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if normalize:
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        if is_mask:
            x = self.pad_tensor_channelwise(
                x, padh, padw, torch.zeros(1), is_mask=True)
        else:
            # for sam. pad after normalize
            if normalize:
                x = self.pad_tensor_channelwise(x, padh, padw, torch.zeros(3))
                # x = x * self.pixel_std + self.pixel_mean

            # for clip. pad before normalize
            else:
                x = self.pad_tensor_channelwise(
                    x, padh, padw, self.clip_pixel_mean)

        return x

    # get point and bbox prompt for training data
    def preprocess_label(self, gt_label, label, select_label):
        select_label[0][0] = gt_label[0]
        point_coords, point_labels, bboxes, categories = [], [], [], []
        point_and_labels = get_points_from_mask(
            select_label[0], top_num=0.5)
        point_coords.append(torch.as_tensor(point_and_labels[0]))
        point_labels.append(torch.as_tensor(point_and_labels[1]))
        bboxes.append(torch.as_tensor(
            get_bboxes_from_mask(select_label[0], offset=5)))
        categories.append(label)

        point_coords = torch.stack(point_coords)
        point_labels = torch.stack(point_labels)
        bboxes = torch.stack(bboxes)

        return select_label, point_coords, point_labels, bboxes, categories

    def make_short_conversation(self, class_name):
        conv = [{
            "from": "human",
            "value": f"<image>\nWhat organ or lesion is shown in this region?<region></region>"
        },
            {
            "from": "gpt",
            "value": f"<SEG> {self.label_mapping[class_name].capitalize()} </SEG>"
        }
        ]

        processed_conv = preprocess_multimodal([conv], self.data_args)

        data_dict = preprocess(
            processed_conv,
            self.tokenizer,
            has_image=True,
            need_region=False)  # 检查原始数据

        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]

        # 构造 sources (注意：现在 item_with_paths 包含了额外的路径key，但 extract_masks_fun 只关心 'conversations')
        sources = {k: v for k, v in item.items() if
                   k != 'imask'}
        sources = [sources]

        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        try:
            label_path = item['label']

            gt_shape = ast.literal_eval(label_path.split('.')[-2])
            allmatrix_sp = sparse.load_npz(label_path)
            label_array = allmatrix_sp.toarray().reshape(gt_shape)  # get label array
            # 根据 class_id 获取对应的 mask
            gt_masks = [label_array[item['class_id']-1, :, :, 0]]

            if 'image' in sources[0]:
                # image_path = self.list_data_dict[i]['image'] # 从原始 item 获取
                image_path = item['image']  # 更直接
                processor = self.data_args.image_processor

                image = cv2.imread(image_path)
                # 处理图像和mask得到image_sam作为imis输入，image_clip作为vlm输入
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = np.array(image_rgb, dtype=np.uint8)
                image_rgb = image_rgb.astype(np.uint8).copy()

                imis_inputs_ori = {'image': image_rgb,
                                   'mask': copy.deepcopy(gt_masks[0])}
                imis_inputs = self.transform_imis(imis_inputs_ori)

                image_imis = imis_inputs['image']
                resize = image_imis.shape[1:]
                mask_imis = imis_inputs['mask']

                clip_inputs_ori = {'image': image_rgb,
                                   'mask': copy.deepcopy(gt_masks[0])}

                clip_inputs = self.transform_clip(clip_inputs_ori)

                # [3,clip_img_size,clip_img_size]
                image_clip = clip_inputs['image']

                if self.data_args.image_aspect_ratio == 'pad':
                    image_clip = processor.preprocess(image_clip, return_tensors='pt')[
                        'pixel_values'][0]
                else:
                    image_clip = processor.preprocess(
                        self.preprocess(torch.from_numpy(image).permute(
                            2, 0, 1).contiguous(), False),
                        return_tensors='pt'
                    )['pixel_values'][0]

                # 处理<image>标志符
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
            else:
                sources = copy.deepcopy(
                    [e["conversations"] for e in sources])

            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('image' in item),
                need_region=False)  # 检查原始数据

            data_dict_short = self.make_short_conversation(item['class'])

            # 转换成torch.tensor
            gt_masks = [process_mask(mask) for mask in gt_masks]

            if isinstance(i, int):
                data_dict = dict(input_ids_seg=data_dict["input_ids"][0],
                                 labels_seg=data_dict["labels"][0],
                                 conversations_seg=data_dict["conversations"],
                                 question_seg=data_dict["question"],
                                 gt_seg=data_dict["gt"],
                                 input_ids_vlm=data_dict_short["input_ids"][0],
                                 labels_vlm=data_dict_short["labels"][0],
                                 conversations_vlm=data_dict_short["conversations"],
                                 question_vlm=data_dict_short["question"],
                                 gt_vlm=data_dict_short["gt"],
                                 )

            if 'image' in item:  # 检查原始数据
                # 经过vision_tower处理后的vlm图像输入
                data_dict['image_clip_seg'] = image_clip
                data_dict['gt_masks_seg'] = gt_masks  # size和原始图像的size一致
            elif self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image_clip_seg'] = torch.zeros(
                    3, crop_size['height'], crop_size['width'])

            data_dict['image_imis_seg'] = image_imis
            data_dict["image_path_seg"] = image_path
            data_dict["proj_only"] = self.proj_only
            data_dict['inference'] = False
            data_dict["answer_type"] = "open"
            data_dict['tokenizer'] = self.tokenizer

            # gt class相关
            data_dict['class_seg'] = [item['class']]

            # imis mask相关
            data_dict['imis_gt_masks_seg'] = mask_imis.unsqueeze(0)

            # 当有mask时，初始化对应大小的label，并保留resize信息
            if len(gt_masks) > 0:
                label = [torch.ones(
                    gt_masks[0].shape[0], gt_masks[0].shape[1]) * self.ignore_label] * len(gt_masks)
                data_dict['label_seg'] = label
                data_dict['resize_seg'] = [
                    resize] * len(gt_masks)

            if not self.proj_only:
                item = self.list_data_dict_vlm[i]
                sources = {k: v for k, v in item.items() if
                           k != 'imask'}
                sources = [sources]
                assert len(
                    sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

                label_path = item['label']

                gt_shape = ast.literal_eval(label_path.split('.')[-2])
                allmatrix_sp = sparse.load_npz(label_path)
                label_array = allmatrix_sp.toarray().reshape(gt_shape)  # get label array
                # 根据 class_id 获取对应的 mask
                gt_masks = [label_array[item['class_id']-1, :, :, 0]]

                if 'image' in sources[0]:
                    # image_path = self.list_data_dict[i]['image'] # 从原始 item 获取
                    image_path = item['image']  # 更直接
                    processor = self.data_args.image_processor

                    image = cv2.imread(image_path)
                    # 处理图像和mask得到image_sam作为imis输入，image_clip作为vlm输入
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_rgb = np.array(image_rgb, dtype=np.uint8)
                    image_rgb = image_rgb.astype(np.uint8).copy()

                    imis_inputs_ori = {'image': image_rgb,
                                       'mask': copy.deepcopy(gt_masks[0])}
                    imis_inputs = self.transform_imis(imis_inputs_ori)

                    image_imis = imis_inputs['image']
                    resize = image_imis.shape[1:]
                    mask_imis = imis_inputs['mask']

                    clip_inputs_ori = {'image': image_rgb,
                                       'mask': copy.deepcopy(gt_masks[0])}

                    clip_inputs = self.transform_clip(clip_inputs_ori)

                    # [3,clip_img_size,clip_img_size]
                    image_clip = clip_inputs['image']

                    if self.data_args.image_aspect_ratio == 'pad':
                        image_clip = processor.preprocess(image_clip, return_tensors='pt')[
                            'pixel_values'][0]
                    else:
                        image_clip = processor.preprocess(
                            self.preprocess(torch.from_numpy(image).permute(
                                2, 0, 1).contiguous(), False),
                            return_tensors='pt'
                        )['pixel_values'][0]

                    # 处理<image>标志符
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                else:
                    sources = copy.deepcopy(
                        [e["conversations"] for e in sources])

                data_dict_ = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=('image' in item),
                    need_region=False)  # 检查原始数据

                # 转换成torch.tensor
                gt_masks = [process_mask(mask) for mask in gt_masks]

                data_dict["input_ids"] = data_dict_["input_ids"][0]
                data_dict["labels"] = data_dict_["labels"][0]
                data_dict["conversations"] = data_dict_["conversations"]
                data_dict["question"] = data_dict_["question"]
                data_dict["gt"] = data_dict_["gt"]

                if 'image' in item:  # 检查原始数据
                    # 经过vision_tower处理后的vlm图像输入
                    data_dict['image_clip'] = image_clip
                    data_dict['gt_masks'] = gt_masks  # size和原始图像的size一致
                elif self.data_args.is_multimodal:
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image_clip'] = torch.zeros(
                        3, crop_size['height'], crop_size['width'])

                data_dict['image_imis'] = image_imis
                data_dict["image_path"] = image_path

                # gt class相关
                data_dict['class'] = [item['class']]

                # imis mask相关
                data_dict['imis_gt_masks'] = mask_imis.unsqueeze(0)

                # 当有mask时，初始化对应大小的label，并保留resize信息
                if len(gt_masks) > 0:
                    label = [torch.ones(
                        gt_masks[0].shape[0], gt_masks[0].shape[1]) * self.ignore_label] * len(gt_masks)
                    data_dict['label'] = label
                    data_dict['resize'] = [
                        resize] * len(gt_masks)

            return data_dict

        except Exception as e:
            print(f"Exception in __getitem__ for index {i}: {e}")
            # traceback.print_exc()
            if i < len(self.list_data_dict) - 1:
                return self.__getitem__(i + 1)


# if __name__ == '__main__':
#     data_path = "/mydata/IMIS_data/BTCV/qa_dataset.json"
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/root/huggingface/clip-vit-large-patch14-336',
#         cache_dir=None,
#         model_max_length=2048,
#         padding_side="right",
#         use_fast=False,
#         legacy=True,
#     )
#     data_args = {"image_folder": '/mydata/IMIS_data/BTCV/',
#                  "sub_image_folder": "/mydata/IMIS_data/BTCV/",
#                  "image_aspect_ratio": 'pad',
#                  "is_multimodal": True,
#                  "mm_use_im_start_end": True,
#                  "data_path": "/mydata/IMIS_data/BTCV/qa_dataset.json",
#                  "image_processor": None
#                  }
#     data_args = types.SimpleNamespace(**data_args)
#     sam_img_size = 256
#     train_dataset = LazySupervisedDataset(
#         data_path, tokenizer, data_args, sam_img_size)
#     train_dataset.__getitem__(10)
