import ast
import time
import types
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import json
import copy
import os
import json
import transformers
from torch.utils.data import Dataset
import random
import torch
from PIL import Image
from monai import data, transforms
import numpy as np
import cv2
import random
from tqdm import tqdm
from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, REGION_TOKEN_INDEX, ADD_OTHERS_TOKENS
from model.mobius import conversation as conversation_lib
from datasets.data_utils import (
    Resize,
    PermuteTransform,
    LongestSidePadding,
    Normalization,
    get_points_from_mask,
    get_bboxes_from_mask
)


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
    conversations_test = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            # record the question and answer
            if sentence['from'] == 'human':
                # 检测有没有<region>，没有就加上
                if need_region and "</region>" not in sentence['value']:
                    sentence['value'] = sentence['value'][:-1] + \
                        "<region></region>" + sentence['value'][-1]
                question.append(sentence['value'].replace(
                    '<im_start><image><im_end>\n', ''))
            else:
                gt.append(sentence['value'])

            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())  # 按照模版合并好的conversation
        conversations_test.append(truncate_conversation(conv.get_prompt()))

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids_test = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations_test], dim=0)

    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        input_ids_test = tokenizer(
            conversations_test,
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
        input_ids_test=input_ids_test,  # 用于rl的 token 序列
        labels=targets,  # mask问题后的目标 token 序列
        conversations=conversations,  # 拼接后的原始prompt
        conversations_test=conversations_test,  # 用于rl的拼接后的原始prompt
        question=question,  # 提取的问题
        gt=gt,  # 提取的真实回答
    )


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


class MedicalDataset_omni(Dataset):
    # for clip
    clip_pixel_mean = (torch.Tensor(
        [0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)
    clip_pixel_std = (torch.Tensor(
        [0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1)*255).clamp(0, 255).to(torch.int)

    def __init__(self, data_path, tokenizer, data_args, num=3):
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.clip_img_size = 336

        self.transform_clip = transforms.Compose(
            [
                Resize(keys=["image"],
                       target_size=(self.clip_img_size, self.clip_img_size)),
                PermuteTransform(keys=["image"], dims=(2, 0, 1)),
                transforms.ToTensord(keys=["image"]),
            ]
        )

        self.list_data_dict = []

        for i, item in enumerate(tqdm(self.data, desc="Loading and validating data")):
            if 'image_path' in item:
                img_path = os.path.join(
                    self.data_args.val_image_folder, item['image_path'])
                if not os.path.exists(img_path):
                    raise FileNotFoundError(
                        f"Image file {img_path} not found.")
                self.list_data_dict.append(item)
            else:
                print("No image field in item")
                continue  # 没有图片字段，跳过？

        print(f"{self.__len__()} samples loaded after validation.")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        entity = self.list_data_dict[idx]
        a, b, c, d = entity.get('option_A'), entity.get(
            'option_B'), entity.get('option_C'), entity.get('option_D')
        answer_list = [a, b]
        if c is not None:
            answer_list.append(c)
        if d is not None:
            answer_list.append(d)
        # question = 'This is a medical Question with several Options, and there is only one correct answer among these options. Please select the correct answer for the question. Remember, you can only select one option. The Question is :' +\
        #     entity['question']+
        question = entity['question'] + f'Here are {len(answer_list)} candidate answers:' + str(
            answer_list) + ' Only return what you think is the correct answer from the candidate answers, do not return any other irrelevant text!'
        answers = entity['gt_answer']
        # 构造convs以匹配输入格式
        convs = [[
            {
                "from": "human",
                "value": "<image>\n" + question
            },
            {
                "from": "gpt",
                "value": answers
            }
        ]]
        sources = preprocess_multimodal(
            copy.deepcopy(convs),
            self.data_args
        )

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image_path' in entity),
            need_region=False)  # 检查原始数据

        img_path = os.path.join(
            self.data_args.val_image_folder, entity['image_path'])

        # 处理image
        processor = self.data_args.image_processor

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = np.array(image_rgb, dtype=np.uint8)
        image_rgb = image_rgb.astype(np.uint8).copy()

        clip_transform_input = {
            'image': image_rgb
        }
        clip_inputs = self.transform_clip(clip_transform_input)
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

        return {
            "image_path": img_path,
            "images_clip": image_clip,
            "question": data_dict['question'],
            "tokenizer": self.tokenizer,
            "input_ids": data_dict["input_ids"][0],
            "answers": data_dict['gt'],
            "answer_list": answer_list,
            "labels": data_dict["labels"][0],
            "answer_type": "closed",
            "entity": entity
        }


class MedicalQADataset(Dataset):
    def __init__(self, data_path, num=64):
        self.data = json.load(open(data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entity = self.data[idx]
        question = entity['question']
        img_path = entity['image_path']

        return {
            "image_path": img_path,
            "question": question,
            "entity": entity
        }


# if __name__ == '__main__':
#     data_path = "/workspace/EviMed/OmniMedVQA/QA_information/Open-access/ACRIMA.json"
#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         '/workspace/EviMed/huggingface/llava-v1.5-7b',
#         cache_dir=None,
#         model_max_length=2048,
#         padding_side="right",
#         use_fast=False,
#         legacy=True,
#     )
#     tokenizer.pad_token = tokenizer.unk_token
#     for i in range(1, 257):
#         ADD_OTHERS_TOKENS.append("<gen_" + str(i) + ">")
#     for token_name in ADD_OTHERS_TOKENS:
#         tokenizer.add_tokens(token_name, special_tokens=True)
#     seg_token_idx = tokenizer(
#         "</SEG>", add_special_tokens=False).input_ids[0]
#     data_args = {"image_folder": '/workspace/EviMed/OmniMedVQA/',
#                  "image_aspect_ratio": 'pad',
#                  "is_multimodal": True,
#                  "mm_use_im_start_end": False,
#                  "data_path": data_path,
#                  "image_processor": None
#                  }
#     data_args = types.SimpleNamespace(**data_args)
#     conversation_lib.default_conversation = conversation_lib.conv_templates[
#         "llava_v1"
#     ]
#     train_dataset = MedicalDataset(
#         data_path, tokenizer, data_args)
#     train_dataset.__getitem__(0)
#     print(train_dataset)
