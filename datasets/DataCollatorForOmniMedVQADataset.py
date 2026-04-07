import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX


def DataCollatorForOmniMedVQADataset(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
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

    image_path_list = []
    images_clip_list = []
    answers_list = []
    questions_list = []
    candidate_list = []
    answer_type_list = []
    for data_dict in list_data_dict:
        image_path_list.append(data_dict.get('image_path', None))
        images_clip_list.append(data_dict.get('images_clip', None))
        answers_list.append(data_dict.get('answers', None))
        questions_list.append(data_dict.get('question', None))
        answer_type_list.append(data_dict.get('answer_type', None))
        candidate_list.append(data_dict.get('answer_list', None))

    final_batch = {
        "image_paths": image_path_list,
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": batch['input_ids'],
        "labels": batch['labels'],
        "attention_masks": batch['attention_mask'],
        "questions_list": questions_list,
        "answers_list": answers_list,
        "seg_flag": seg_flag,
        "inference": inference,
        "answer_type": answer_type_list,
        "candidate_list": candidate_list,
    }
    return final_batch
