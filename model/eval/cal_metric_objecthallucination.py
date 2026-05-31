import argparse
import json
import collections
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from tabulate import tabulate
from glossary import *
import os
import difflib
from tqdm import tqdm

import warnings
warnings.simplefilter('ignore')


def parse_option():
    parser = argparse.ArgumentParser(
        'Evaluation for LLaVA Generated Outputs', add_help=False)
    # parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument(
        '--pred', type=str, default="/workspace/EviMed/IMIS_tests/imis-stage1/ObjectHallucination_new_noregion/ObjectHallucinationBenchmark_AllTypes_NoRegion.jsonl", help='path to prediction file', )
    parser.add_argument(
        '--save', type=str, default="/workspace/EviMed/IMIS_tests/imis-stage1/ObjectHallucination_new_noregion/", help='save path', )
    parser.add_argument('--candidate_set', type=str,
                        default="/workspace/EviMed/MedPLIB_old_fortest/datasets/short_prompt_constant.json", help='path to prediction file', )
    args, unparsed = parser.parse_known_args()
    return args


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    # Return the index of the most similar string
    return most_similar_index, most_similar_str


def evaluate(test_dict_lst, args, criterion=None):
    print('pred file', args.pred)
    print('candidate_set', args.candidate_set)

    closed_scores_modal_dict = {}
    closed_scores_modal_dict_exact = {}

    closed_scores = collections.defaultdict(list)
    closed_scores_exact = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    f1_scores = collections.defaultdict(list)
    open_hit_scores = collections.defaultdict(list)

    eval_open = False
    eval_closed = False
    open_cnt = 0
    closed_cnt = 0
    for item in tqdm(test_dict_lst):
        gt_value = item['gt'].lower()
        pred_value = item['text'].lower()

        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        if item['answer_type'].lower() in ['open', 'other', 'number']:
            eval_open = True
            # for open-ended question
            # if gt_value in pred_value:
            #     hit = 1.0
            # else:
            #     hit = 0.0
            # open_hit_scores['hit'].append(hit)

            if args.candidate_set is not None:
                candidate_set = item['candidate_list']
                open_hit_scores['hit'].append(calculate_appearance_with_normalization(
                    pred_value, gt_value, candidate_set))
                open_hit_scores['q_id'].append(item['question_id'])

            exact_scores['hit'].append(
                calculate_exactmatch(pred_value, gt_value))
            exact_scores['q_id'].append(item['question_id'])

            # import pdb; pdb.set_trace()

            f1_score, precision, recall = calculate_f1score(
                pred_value, gt_value)
            f1_scores['f1'].append(f1_score)
            f1_scores['precision'].append(precision)
            f1_scores['recall'].append(recall)
            f1_scores['q_id'].append(item['question_id'])

            # if isinstance(f1_scores['hit'][-1], str):
            #     # import pdb; pdb.set_trace()

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))

            bleu_scores['q_id'].append(item['question_id'])
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)
            open_cnt += 1

        elif item['answer_type'].lower() in ["yes/no", 'closed']:
            type = item['data_type']
            if type not in closed_scores_modal_dict:
                closed_scores_modal_dict[type] = collections.defaultdict(list)
                closed_scores_modal_dict_exact[type] = collections.defaultdict(
                    list)
            eval_closed = True
            # for close-ended question (Yes/No)
            closed_scores_modal_dict[type]['q_id'].append(item['question_id'])
            closed_scores_modal_dict_exact[type]['q_id'].append(
                item['question_id'])

            # calculate exact match
            if 'yes' in pred_value or 'no' in pred_value:
                if gt_value in pred_value:
                    closed_scores_modal_dict_exact[type]['hit'].append(1)
                else:
                    closed_scores_modal_dict_exact[type]['hit'].append(0)
            else:
                closed_scores_modal_dict_exact[type]['hit'].append(0)

            # find the most similar answer in the candidate list
            try:
                answer_list = item['candidate_list']
                norm_answer_list = []
                for answer in answer_list:
                    norm_answer_list.append(normalize_word(answer))

                idx, sim_str = find_most_similar_index(
                    norm_answer_list, pred_value)

                if gt_value == norm_answer_list[idx]:
                    closed_scores_modal_dict[type]['hit'].append(1)
                else:
                    closed_scores_modal_dict[type]['hit'].append(0)
                closed_cnt += 1
            except Exception as e:
                print("Error in finding most similar answer:", e)
                continue

    print('open_cnt', open_cnt)
    print('closed_cnt', closed_cnt)
    mean_lst = []
    mean_lst_exact = []
    closed_scores_modal_dict_res = {}
    closed_scores_modal_dict_res_exact = {}
    if eval_closed:
        for k in closed_scores_modal_dict.keys():
            closed_score = sum(closed_scores_modal_dict[k]['hit']) / len(
                closed_scores_modal_dict[k]['hit']) if len(closed_scores_modal_dict[k]['hit']) != 0 else 0.0
            closed_scores_modal_dict_res[k] = round(closed_score*100, 2)
            mean_lst.append(closed_score)
        for k in closed_scores_modal_dict_exact.keys():
            closed_score_exact = sum(closed_scores_modal_dict_exact[k]['hit']) / len(
                closed_scores_modal_dict_exact[k]['hit']) if len(closed_scores_modal_dict_exact[k]['hit']) != 0 else 0.0
            closed_scores_modal_dict_res_exact[k] = round(
                closed_score_exact*100, 2)
            mean_lst_exact.append(closed_score_exact)
    else:
        closed_score = 0.0
        closed_score_exact = 0.0

    per_type_acc = tabulate(closed_scores_modal_dict_res.items(), headers=[
                            'Type', 'Accuracy'], tablefmt='github')
    per_type_acc_exact = tabulate(closed_scores_modal_dict_res_exact.items(), headers=[
                                  'Type', 'Accuracy'], tablefmt='github')
    all = tabulate(
        [
            ['exact match acc score', (sum(
                mean_lst_exact)/len(mean_lst_exact))*100],
            ['soft acc score', (sum(mean_lst)/len(mean_lst))*100],
            ['closed cnt', closed_cnt]
        ],
        headers=['Metric', 'Performance'],
        tablefmt='github'
    )

    print(all)
    print("\n")
    print(per_type_acc_exact)
    print("\n")
    print(per_type_acc)

    return all, per_type_acc, per_type_acc_exact


if __name__ == '__main__':
    args = parse_option()

    # gt = json.load(open(args.gt, 'r'))
    test_dict_lst = load_jsonl(args.pred)

    # gt_ids = [item['id'] for item in gt]
    # pred_ids = [item['question_id'] for item in pred]
    # # import pdb; pdb.set_trace()
    # assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    # perform evaluation
    results, per_type_acc, per_type_acc_exact = evaluate(test_dict_lst, args)

    # 写入txt文件
    with open(os.path.join(args.save, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(results+"\n"+"\n")
        f.write(per_type_acc+"\n"+"\n")
        f.write(per_type_acc_exact)

    print("保存成功！")
