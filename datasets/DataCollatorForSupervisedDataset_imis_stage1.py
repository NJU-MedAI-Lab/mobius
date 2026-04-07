import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX


def DataCollatorForSupervisedDataset_stage1(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    input_ids, labels = tuple([instance[key] for instance in list_data_dict]
                              for key in ("input_ids", "labels"))
    # 将所有的input_ids扩充成相同长度s
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    # 将所有的labels扩充成相同长度
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    # 将所有的input_ids和labels截断到相同长度，tokenizer.model_max_length
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        # 指示input_ids中哪些token是有效的
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    seg_flag = False
    max_mask_nums = max([len(instance['masks'])
                        for instance in list_data_dict])
    # fullfill mask instances according to the max_mask_nums in this batch
    masks = []
    label_list = []
    valid_mask_bool = []
    if max_mask_nums > 0:
        seg_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['masks']) > 0:
                masks.extend(instance['masks'])
                label_list.extend(instance['label'])
                valid_mask_bool.append([True] * len(instance['masks']))
            else:
                valid_mask_bool.append([])

    batch['masks'] = masks
    batch['valid_mask_bool'] = valid_mask_bool
    batch['label_list'] = label_list

    region_masks = []
    valid_region_masks_bool = []
    max_region_masks_nums = max(
        [len(instance['region_masks']) for instance in list_data_dict])

    rp_flag = False
    if max_region_masks_nums > 0:
        rp_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['region_masks']) > 0:
                region_masks.extend(instance['region_masks'])
                valid_region_masks_bool.append(
                    [torch.ones(1).bool()] * len(instance['region_masks']))
            else:
                valid_region_masks_bool.append([torch.zeros(1).bool()])

    image_path_list = []
    images_imis_list = []
    images_clip_list = []
    conversation_list = []
    # resize_list = []
    # label_list = []
    questions_list = []
    gts_list = []
    sampled_classes_list = []
    offset_list = [0]
    answer_type_list = []
    gt_imis_labels = []
    target_list = []
    cnt = 0
    for data_dict in list_data_dict:
        image_path_list.append(data_dict.get('image_path', None))
        images_imis_list.append(data_dict.get('image_imis', None))
        images_clip_list.append(data_dict.get('image_clip', None))
        conversation_list.extend(data_dict.get('conversations', None))
        # label_list.append(data_dict.get('label', torch.tensor([])))
        # resize_list.append(data_dict.get('resize', None))
        questions_list.append(data_dict.get('question', None))
        gts_list.append(data_dict.get('gt', None))
        sampled_classes_list.append(data_dict.get('sampled_classes', None))
        cnt += len(data_dict.get('conversations', None))
        offset_list.append(cnt)
        answer_type_list.append(data_dict.get('answer_type', None))
        gt_imis_labels.append(data_dict.get('gt_imis_label', None))

    final_batch = {
        "image_paths": image_path_list,
        "images_imis": torch.stack(images_clip_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": batch['input_ids'],
        "labels": batch['labels'],
        "attention_masks": batch['attention_mask'],
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "answers_list": gts_list,
        "sampled_classes_list": sampled_classes_list,
        "conversation_list": conversation_list,
        "seg_flag": seg_flag,
        "inference": inference,
        "region_masks": region_masks,
        "valid_region_masks_bool": valid_region_masks_bool,
        "answer_type": answer_type_list,
    }
    return final_batch
