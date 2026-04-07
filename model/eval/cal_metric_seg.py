import json
import os
from tqdm import tqdm
from tabulate import tabulate

import warnings
warnings.simplefilter('ignore')

data_path = "/workspace/EviMed/MedPLIB_tests/MedPLIB-7b-2e/Text_Instructed_Seg/Stage4_Seg.jsonl"
save_path = "/workspace/EviMed/MedPLIB_tests/MedPLIB-7b-2e/Text_Instructed_Seg/metric.txt"


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


test_results = load_jsonl(data_path)

modality_dict = {}
class_dict = {}
total_ious = []
total_dices = []

pbar = tqdm(test_results)

for item in pbar:
    modality = item['modality']
    class_name = item['class']
    iou = item['iou']
    dice = item['dice']

    if modality not in modality_dict.keys():
        modality_dict[modality] = {'iou': [], 'dice': [], 'count': 0}
    if class_name not in class_dict.keys():
        class_dict[class_name] = {'iou': [], 'dice': [], 'count': 0}

    modality_dict[modality]['iou'].append(iou)
    modality_dict[modality]['dice'].append(dice)
    modality_dict[modality]['count'] += 1

    class_dict[class_name]['iou'].append(iou)
    class_dict[class_name]['dice'].append(dice)
    class_dict[class_name]['count'] += 1

    total_ious.append(iou)
    total_dices.append(dice)

miou = round(sum(total_ious) / len(total_ious)*100, 4)
mdice = round(sum(total_dices) / len(total_dices)*100, 4)

per_modality_results = []
per_class_results = []

for key in modality_dict.keys():
    avg_iou = round(
        sum(modality_dict[key]['iou']) / modality_dict[key]['count']*100, 4)
    avg_dice = round(
        sum(modality_dict[key]['dice']) / modality_dict[key]['count']*100, 4)
    per_modality_results.append(
        [key, avg_iou, avg_dice, modality_dict[key]['count']])

for key in class_dict.keys():
    avg_iou = round(
        sum(class_dict[key]['iou']) / class_dict[key]['count']*100, 4)
    avg_dice = round(
        sum(class_dict[key]['dice']) / class_dict[key]['count']*100, 4)
    per_class_results.append(
        [key, avg_iou, avg_dice, class_dict[key]['count']])

per_modality_metric = tabulate(per_modality_results, headers=[
    'Modality', 'mIoU', 'mDice', 'Count'], tablefmt='github')

per_class_metric = tabulate(per_class_results, headers=[
    'Class', 'mIoU', 'mDice', 'Count'], tablefmt='github')

avg_metric = tabulate([[miou, mdice, len(test_results)]], headers=[
    'mIoU', 'mDice', 'Count'], tablefmt='github')

with open(save_path, 'w', encoding='utf-8') as f:
    f.write("Overall Metrics:\n")
    f.write(avg_metric + "\n\n")
    f.write("Per Modality Metrics:\n")
    f.write(per_modality_metric + "\n\n")
    f.write("Per Class Metrics:\n")
    f.write(per_class_metric + "\n")
