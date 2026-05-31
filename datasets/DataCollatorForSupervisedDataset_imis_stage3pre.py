import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX


def DataCollatorForSupervisedDataset_stage3pre(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    proj_only = list_data_dict[0].get('proj_only', False)
    input_ids_seg, labels_seg, input_ids_vlm, labels_vlm = tuple([instance[key] for instance in list_data_dict]
                                                                 for key in ("input_ids_seg", "labels_seg", "input_ids_vlm", "labels_vlm"))
    if not proj_only:
        input_ids, labels = tuple([instance[key] for instance in list_data_dict]
                                  for key in ("input_ids", "labels"))
    # 将所有的input_ids扩充成相同长度s
    input_ids_seg = torch.nn.utils.rnn.pad_sequence(
        input_ids_seg,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    # 将所有的labels扩充成相同长度
    labels_seg = torch.nn.utils.rnn.pad_sequence(labels_seg,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
    # 将所有的input_ids扩充成相同长度s
    input_ids_vlm = torch.nn.utils.rnn.pad_sequence(
        input_ids_vlm,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    # 将所有的labels扩充成相同长度
    labels_vlm = torch.nn.utils.rnn.pad_sequence(labels_vlm,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

    # 将所有的input_ids和labels截断到相同长度，tokenizer.model_max_length
    input_ids_seg = input_ids_seg[:, :tokenizer.model_max_length]
    labels_seg = labels_seg[:, :tokenizer.model_max_length]
    input_ids_vlm = input_ids_vlm[:, :tokenizer.model_max_length]
    labels_vlm = labels_vlm[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids_seg=input_ids_seg,
        labels_seg=labels_seg,
        # 指示input_ids中哪些token是有效的
        attention_mask_seg=input_ids_seg.ne(tokenizer.pad_token_id),
        input_ids_vlm=input_ids_vlm,
        labels_vlm=labels_vlm,
        attention_mask_vlm=input_ids_vlm.ne(tokenizer.pad_token_id),
    )

    seg_flag = False
    max_mask_nums = max([len(instance['gt_masks_seg'])
                         for instance in list_data_dict])
    # fullfill mask instances according to the max_mask_nums in this batch
    masks = []
    label_list = []
    resize_list = []
    valid_mask_bool = []
    if max_mask_nums > 0:
        seg_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['gt_masks_seg']) > 0:
                masks.extend(instance['gt_masks_seg'])
                label_list.extend(instance['label_seg'])
                resize_list.extend(instance['resize_seg'])
                valid_mask_bool.append([True] * len(instance['gt_masks_seg']))
            else:
                valid_mask_bool.append([])

    batch['masks_seg'] = masks
    batch['valid_mask_bool_seg'] = valid_mask_bool
    batch['label_list_seg'] = label_list
    batch['resize_list_seg'] = resize_list

    if not proj_only:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        # 将所有的labels扩充成相同长度
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :tokenizer.model_max_length]
        labels = labels[:, :tokenizer.model_max_length]

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["attention_mask"] = input_ids.ne(tokenizer.pad_token_id)

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

    image_path_seg_list = []
    images_imis_seg_list = []
    images_clip_seg_list = []
    conversation_seg_list = []
    questions_seg_list = []
    gts_seg_list = []
    conversation_vlm_list = []
    questions_vlm_list = []
    gts_vlm_list = []
    imis_gt_masks_seg = []
    image_path_list = []
    images_imis_list = []
    images_clip_list = []
    conversation_list = []
    questions_list = []
    gts_list = []
    imis_gt_masks = []
    offset_list = [0]
    answer_type_list = []
    cnt = 0
    for data_dict in list_data_dict:
        image_path_seg_list.append(data_dict.get('image_path_seg', None))
        images_imis_seg_list.append(data_dict.get('image_imis_seg', None))
        images_clip_seg_list.append(data_dict.get('image_clip_seg', None))
        conversation_seg_list.extend(data_dict.get('conversations_seg', None))
        questions_seg_list.append(data_dict.get('question_seg', None))
        gts_seg_list.append(data_dict.get('gt_seg', None))
        conversation_vlm_list.extend(data_dict.get('conversations_vlm', None))
        questions_vlm_list.append(data_dict.get('question_vlm', None))
        gts_vlm_list.append(data_dict.get('gt_vlm', None))
        imis_gt_masks_seg.extend(data_dict.get('imis_gt_masks_seg', []))
        cnt += len(data_dict.get('conversations_seg', None))
        offset_list.append(cnt)
        answer_type_list.append(data_dict.get('answer_type', None))
        if not proj_only:
            image_path_list.append(data_dict.get('image_path', None))
            images_imis_list.append(data_dict.get('image_imis', None))
            images_clip_list.append(data_dict.get('image_clip', None))
            conversation_list.extend(data_dict.get('conversations', None))
            questions_list.append(data_dict.get('question', None))
            gts_list.append(data_dict.get('gt', None))
            imis_gt_masks.extend(data_dict.get('imis_gt_masks', []))

    if proj_only:
        final_batch = {
            "image_paths": image_path_seg_list,
            "images_imis": torch.stack(images_imis_seg_list, dim=0),
            "images_clip": torch.stack(images_clip_seg_list, dim=0),
            "input_ids_seg": batch['input_ids_seg'],
            "labels_seg": batch['labels_seg'],
            "attention_masks_seg": batch['attention_mask_seg'],
            "input_ids": batch['input_ids_vlm'],
            "labels": batch['labels_vlm'],
            "attention_masks": batch['attention_mask_vlm'],
            "masks_list_seg": batch['masks_seg'],
            "label_list_seg": batch['label_list_seg'],
            "resize_list_seg": batch['resize_list_seg'],
            "offset": torch.LongTensor(offset_list),
            "questions_list_seg": questions_seg_list,
            "answers_list_seg": gts_seg_list,
            "conversation_list_seg": conversation_seg_list,
            "questions_list": questions_vlm_list,
            "answers_list": gts_vlm_list,
            "conversation_list": conversation_vlm_list,
            "imis_gt_masks": torch.stack(imis_gt_masks_seg, dim=0),
            "answer_type": answer_type_list,
            "seg_flag": seg_flag,
            "inference": inference,
        }
    else:
        final_batch = {
            "image_paths_seg": image_path_seg_list,
            "images_imis_seg": torch.stack(images_imis_seg_list, dim=0),
            "images_clip_seg": torch.stack(images_clip_seg_list, dim=0),
            "input_ids_seg": batch['input_ids_seg'],
            "labels_seg": batch['labels_seg'],
            "attention_masks_seg": batch['attention_mask_seg'],
            "input_ids_vlm": batch['input_ids_vlm'],
            "labels_vlm": batch['labels_vlm'],
            "attention_masks_vlm": batch['attention_mask_vlm'],
            "masks_list_seg": batch['masks_seg'],
            "label_list_seg": batch['label_list_seg'],
            "resize_list_seg": batch['resize_list_seg'],
            "offset": torch.LongTensor(offset_list),
            "questions_list_seg": questions_seg_list,
            "answers_list_seg": gts_seg_list,
            "conversation_list_seg": conversation_seg_list,
            "questions_list_vlm": questions_vlm_list,
            "answers_list_vlm": gts_vlm_list,
            "conversation_list_vlm": conversation_vlm_list,
            "imis_gt_masks_seg": torch.stack(imis_gt_masks_seg, dim=0),
            "answer_type": answer_type_list,
            "seg_flag": seg_flag,
            "inference": inference,
            # vlm->sam data
            "image_paths": image_path_list,
            "images_imis": torch.stack(images_imis_list, dim=0),
            "images_clip": torch.stack(images_clip_list, dim=0),
            "input_ids": batch['input_ids'],
            "labels": batch['labels'],
            "attention_masks": batch['attention_mask'],
            "masks_list": batch['masks'],
            "label_list": batch['label_list'],
            "resize_list": batch['resize_list'],
            "questions_list": questions_list,
            "answers_list": gts_list,
            "imis_gt_masks": torch.stack(imis_gt_masks, dim=0),
            "conversation_list": conversation_list,
        }
    return final_batch
