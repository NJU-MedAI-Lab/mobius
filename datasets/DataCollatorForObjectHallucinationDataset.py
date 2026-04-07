import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX


def DataCollatorForObjectHallucinationDataset(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    seg_flag = list_data_dict[0].get('seg_flag', False)
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

    region_masks = []
    valid_region_masks_bool = []
    max_region_masks_nums = max(
        [len(instance['region_masks']) for instance in list_data_dict])

    if max_region_masks_nums > 0:
        for idx, instance in enumerate(list_data_dict):
            if len(instance['region_masks']) > 0:
                region_masks.extend(instance['region_masks'])
                valid_region_masks_bool.append(
                    [torch.ones(1).bool()] * len(instance['region_masks']))
            else:
                valid_region_masks_bool.append([torch.zeros(1).bool()])

    image_path_list = []
    images_clip_list = []
    answers_list = []
    questions_list = []
    candidate_list = []
    answer_type_list = []
    data_type_list = []
    for data_dict in list_data_dict:
        image_path_list.append(data_dict.get('image_path', None))
        images_clip_list.append(data_dict.get('images_clip', None))
        answers_list.append(data_dict.get('answers', None))
        questions_list.append(data_dict.get('question', None))
        candidate_list.append(data_dict.get('answer_list', None))
        answer_type_list.append(data_dict.get('answer_type', None))
        data_type_list.append(data_dict.get('type', 'unknown'))

    if seg_flag:
        input_ids_seg, labels_seg = tuple([instance[key] for instance in list_data_dict]
                                          for key in ("input_ids_seg", "labels_seg"))
        # 将所有的input_ids扩充成相同长度s
        input_ids_seg = torch.nn.utils.rnn.pad_sequence(
            input_ids_seg,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        # 将所有的labels扩充成相同长度
        labels_seg = torch.nn.utils.rnn.pad_sequence(labels_seg,
                                                     batch_first=True,
                                                     padding_value=IGNORE_INDEX)
        # 将所有的input_ids和labels截断到相同长度，tokenizer.model_max_length
        input_ids_seg = input_ids_seg[:, :tokenizer.model_max_length]
        labels_seg = labels_seg[:, :tokenizer.model_max_length]
        batch_seg = dict(
            input_ids=input_ids_seg,
            labels=labels_seg,
            # 指示input_ids中哪些token是有效的
            attention_mask=input_ids_seg.ne(tokenizer.pad_token_id),
        )

        answers_list_seg = []
        questions_list_seg = []
        images_imis_list = []
        for data_dict in list_data_dict:
            answers_list_seg.append(data_dict.get('answers_seg', None))
            questions_list_seg.append(data_dict.get('question_seg', None))
            images_imis_list.append(data_dict.get('images_imis', None))

        final_batch = {
            "image_paths": image_path_list,
            "images_clip": torch.stack(images_clip_list, dim=0),
            "images_imis": torch.stack(images_imis_list, dim=0),
            "input_ids": batch['input_ids'],
            "labels": batch['labels'],
            "attention_masks_seg": batch_seg['attention_mask'],
            "input_ids_seg": batch_seg['input_ids'],
            "labels_seg": batch_seg['labels'],
            "attention_masks": batch['attention_mask'],
            "questions_list": questions_list,
            "answers_list": answers_list,
            "questions_list_seg": questions_list_seg,
            "answers_list_seg": answers_list_seg,
            "region_masks": region_masks,
            "valid_region_masks_bool": valid_region_masks_bool,
            "seg_flag": seg_flag,
            "inference": inference,
            "answer_type": answer_type_list,
            "data_type": data_type_list,
            "candidate_list": candidate_list,
        }

    else:
        final_batch = {
            "image_paths": image_path_list,
            "images_clip": torch.stack(images_clip_list, dim=0),
            "images_imis": torch.stack(images_clip_list, dim=0),
            "input_ids": batch['input_ids'],
            "labels": batch['labels'],
            "attention_masks": batch['attention_mask'],
            "questions_list": questions_list,
            "answers_list": answers_list,
            "region_masks": region_masks,
            "valid_region_masks_bool": valid_region_masks_bool,
            "seg_flag": seg_flag,
            "inference": inference,
            "answer_type": answer_type_list,
            "data_type": data_type_list,
            "candidate_list": candidate_list,
        }
    return final_batch
