import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX


def DataCollatorForSupervisedDataset_test(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    input_ids, labels = tuple([instance[key] for instance in list_data_dict]
                              for key in ("input_ids", "labels"))
    # 将所有的input_ids扩充成相同长度
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
    max_mask_nums = max([len(instance['gt_masks'])
                        for instance in list_data_dict])
    # fullfill mask instances according to the max_mask_nums in this batch
    masks = []
    label_list = []
    resize_list = []
    valid_mask_bool = []
    if max_mask_nums > 0:
        seg_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['gt_masks']) > 0:
                masks.extend(instance['gt_masks'])
                label_list.extend(instance['label'])
                resize_list.extend(instance['resize'])
                valid_mask_bool.append([True] * len(instance['gt_masks']))
            else:
                valid_mask_bool.append([])

    batch['masks'] = masks
    batch['valid_mask_bool'] = valid_mask_bool
    batch['label_list'] = label_list
    batch['resize_list'] = resize_list

    image_path_list = []
    images_imis_list = []
    images_clip_list = []
    conversation_list = []
    # resize_list = []
    # label_list = []
    questions_list = []
    gts_list = []
    imis_gt_masks = []
    offset_list = [0]
    answer_type_list = []
    gt_imis_labels = []
    class_list = []
    modality_list = []
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
        imis_gt_masks.extend(data_dict.get('imis_gt_masks', []))
        cnt += len(data_dict.get('conversations', None))
        offset_list.append(cnt)
        answer_type_list.append(data_dict.get('answer_type', None))
        gt_imis_labels.append(data_dict.get('gt_imis_label', None))
        class_list.extend(data_dict.get('class', None))
        modality_list.append(data_dict.get('modality', None))

    final_batch = {
        "image_paths": image_path_list,
        "images_imis": torch.stack(images_imis_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": batch['input_ids'],
        "labels": batch['labels'],
        "attention_masks": batch['attention_mask'],
        "masks_list": batch['masks'],
        "label_list": batch['label_list'],
        "resize_list": batch['resize_list'],
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "answers_list": gts_list,
        "imis_gt_masks": torch.stack(imis_gt_masks, dim=0),
        "conversation_list": conversation_list,
        "seg_flag": seg_flag,
        "inference": inference,
        "answer_type": answer_type_list,
        "class_list": class_list,
        "modality_list": modality_list,
    }
    return final_batch
