import glob
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, ADD_OTHERS_TOKENS)
from datasets import MedicalDataset_omni, DataCollatorForOmniMedVQADataset, MedicalDataset_hallu, DataCollatorForObjectHallucinationDataset
from model.mobius import conversation as conversation_lib
from model.Mobius import MobiusForCausalLM
from torch.utils.data import Dataset, Subset
import cv2
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model
import transformers
import tqdm
import torch
import numpy as np
import deepspeed
import json
import shortuuid
import random
import math
import types
from functools import partial
import time
import shutil
import argparse
import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


def parse_args(args):
    parser = argparse.ArgumentParser(description="Mobius Testing")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/workspace/EviMed/huggingface_models/llava-v1.5-7b"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--pretrain_mm_mlp_adapter", default=None, type=str)
    parser.add_argument("--pretrain_sam", default=None, type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--sam_img_size", default=1024,
                        type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument(
        "--vision_tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument('--imis_sam_checkpoint', type=str,
                        default="/workspace/EviMed/huggingface/IMISNet-B.pth")

    # data setting
    parser.add_argument('--image_folder', type=str, default='/tmp/v2_mnt/HCG/huangxiaoshuang/SAMed2D_v1',
                        help='Path to the folder containing images.')
    parser.add_argument('--image_aspect_ratio', type=str,
                        default='pad', help='How to handle image aspect ratio.')
    parser.add_argument('--is_multimodal', action='store_true',
                        default=True, help='Whether to use multimodal data.')
    parser.add_argument('--val_data_path', type=str, default='/tmp/v2_mnt/HCG/huangxiaoshuang/med-vqa-dataset/ImageClef-2019-VQA-Med/test_llavaformat_oneturn_open.json',
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--answer_type', type=str,
                        default='closed', help='answer_type.')
    parser.add_argument('--benchmark_name', type=str,
                        default='omniMedVQA', help='benchmark_name.')

    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--iou_loss_weight", default=2.0, type=float)
    parser.add_argument("--focal_loss_weight", default=2.0, type=float)
    parser.add_argument("--vision_pretrained",
                        default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=768, type=int)
    parser.add_argument("--train_mask_decoder",
                        action="store_true", default=False)
    parser.add_argument("--use_mm_start_end",
                        action="store_true", default=True)

    # -----------------infer-------------------
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument('--cpu_only', action='store_true',
                        default=False, help='')
    parser.add_argument('--vis_mask', action='store_true',
                        default=False, help='')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default='')

    # ---------------rp sampler-----------
    parser.add_argument("--region_fea_adapter",
                        action="store_true", default=False)
    parser.add_argument("--region_geo_sampler",
                        action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default='max', type=str)

    return parser.parse_args(args)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def test_randomness():
    # Python random
    print("Python random:")
    print([random.randint(1, 100) for _ in range(5)])

    # NumPy random
    print("\nNumPy random:")
    print(np.random.rand(5))

    # PyTorch random
    print("\nPyTorch random:")
    print(torch.rand(5))

    # PyTorch layer weights
    linear = torch.nn.Linear(10, 5)
    print("PyTorch Linear weights:")
    print(linear.weight)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def main(args):
    args = parse_args(args)
    deepspeed.init_distributed(dist_backend='nccl')
    set_seed(42)
    if args.local_rank == 0:
        test_randomness()
        print('val_data_path is', args.val_data_path)
    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    args.seg_token_idx = tokenizer(
        "</SEG>", add_special_tokens=False).input_ids[0]

    conversation_lib.default_conversation = conversation_lib.conv_templates['v1']

    model_args = vars(args)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_args['test_only'] = True

    # ckpt = {}
    # bin_files = glob.glob(os.path.join(
    #     args.pretrain_sam, '*.bin'))
    # for file in bin_files:
    #     part_ckpt = torch.load(file, map_location='cpu')
    #     ckpt.update(part_ckpt)

    # for k, v in ckpt.items():
    #     if 'mm_projector' in k:
    #         print(
    #             f"[DEBUG] mm projector ori {k} mean: {v.mean()} std: {v.std()}")
    # load LLM and tower weights
    # load sam and text_hidden_fcs weights
    model = MobiusForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=False, ignore_mismatched_sizes=False, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_imis_modules(model.get_model().config, vars(args))
    # vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if not args.cpu_only:
        model.to(dtype=torch_dtype, device=args.local_rank)

    # for name, param in model.named_parameters():
    #     if 'mm_projector' in name:
    #         print(
    #             f"[DEBUG] loaded mm projector {name} mean: {param.data.mean()} std: {param.data.std()}")

    if args.local_rank == 0:
        for name, param in model.named_parameters():
            param.requires_grad = False
            # print(f"Parameter Name: {name}, Shape: {param.shape}, grad: {param.requires_grad}")
            # print(param)
    # --------------show trainable params---------------

    def count_parameters(model):
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    trainable_params, total_params = count_parameters(model)
    trainable_params_percentage = trainable_params / total_params * 100

    if args.local_rank == 0:
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Total Parameters: {total_params}")
        print(
            f"Trainable Parameters Percentage: {trainable_params_percentage:.7f}%")
    # -------------------------------------------------

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    data_args = {"val_image_folder": args.image_folder,
                 "image_aspect_ratio": args.image_aspect_ratio,
                 "is_multimodal": args.is_multimodal,
                 "mm_use_im_start_end": args.use_mm_start_end,
                 "image_processor": vision_tower.image_processor
                 }

    data_args = types.SimpleNamespace(**data_args)

    # val_dataset = LazySupervisedDataset(
    #     args.val_data_path, tokenizer, data_args, args.sam_img_size, test_mode=True)
    if args.benchmark_name == 'omniMedVQA':
        val_dataset = MedicalDataset_omni(
            args.val_data_path, tokenizer, data_args)
    elif args.benchmark_name == 'hallucination' or args.benchmark_name == 'hallucination_region':
        val_dataset = MedicalDataset_hallu(
            args.val_data_path, tokenizer, data_args)

    val_indices = get_chunk(range(len(val_dataset)),
                            args.num_chunks, args.chunk_idx)
    val_dataset = Subset(val_dataset, val_indices)

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        if args.benchmark_name == 'omniMedVQA':
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                # sampler=val_sampler,
                drop_last=False,
                collate_fn=partial(
                    DataCollatorForOmniMedVQADataset,
                    inference=True,
                ),
            )
        elif args.benchmark_name == 'hallucination' or args.benchmark_name == 'hallucination_region':
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                # sampler=val_sampler,
                drop_last=False,
                collate_fn=partial(
                    DataCollatorForObjectHallucinationDataset,
                    inference=True,
                ),
            )
    print(f"Validation dataset size: {len(val_dataset)}")

    model.eval()
    if args.benchmark_name == 'hallucination_region':
        validate_region_vqa(val_loader, model, 0, args,
                            tokenizer)
    else:
        validate_vqa(val_loader, model, 0, args,
                     tokenizer)
    exit()


def validate_vqa(val_loader, model_engine, epoch, args, tokenizer):
    model_name = args.version.split('/')[-2]
    file_name = args.val_data_path.split('/')[-2]
    # answers_file = os.path.join('./runs', model_name,'infer_res', file_name + '.jsonl')

    answers_file = args.answers_file

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if args.local_rank == 0 and os.path.exists(answers_file):
        os.remove(answers_file)
    ans_file = open(answers_file, "a")

    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
        # for input_dict in val_loader:

        # if idx > 20:
        #     break

        if not args.cpu_only:
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images_clip"] = input_dict["images_clip"].float()

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_masks'][:, :indices[1][-1]+1]
        candidate_list = input_dict['candidate_list'][0]
        # region_masks = input_dict['region_masks']
        # valid_region_masks_bool = input_dict['valid_region_masks_bool']
        # if args.local_rank == 3:
        # if input_dict.get('region_masks', None) is not None:
        #     print(input_ids)
        #     print(input_dict.get('valid_region_masks_bool', None))
        with torch.no_grad():
            output_ids = model_engine.generate(
                input_ids,
                images=input_dict['images_clip'],
                region_masks=input_dict.get('region_masks', None),
                valid_region_masks_bool=input_dict.get(
                    'valid_region_masks_bool', None),
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=20,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "image_path": input_dict['image_paths'][0],
                                   "prompt": input_dict['questions_list'][0][0],
                                   "gt": input_dict['answers_list'][0][0],
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "answer_type": args.answer_type,
                                   "data_type": input_dict['data_type'][0] if args.benchmark_name == 'hallucination' else 'omniMedVQA',
                                   "candidate_list": candidate_list,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        idx += 1
    ans_file.close()


def validate_region_vqa(val_loader, model_engine, epoch, args, tokenizer):
    model_name = args.version.split('/')[-2]
    file_name = args.val_data_path.split('/')[-2]
    # answers_file = os.path.join('./runs', model_name,'infer_res', file_name + '.jsonl')

    answers_file = args.answers_file

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if args.local_rank == 0 and os.path.exists(answers_file):
        os.remove(answers_file)
    ans_file = open(answers_file, "a")

    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
        if not args.cpu_only:
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images_imis"] = input_dict["images_imis"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_imis"] = input_dict["images_imis"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images_imis"] = input_dict["images_imis"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_masks'][:, :indices[1][-1]+1]
        candidate_list = input_dict['candidate_list'][0]
        seg_flag = input_dict.get("seg_flag", False)
        with torch.no_grad():
            if seg_flag:
                output_ids = model_engine.evaluate_sam_vlm(
                    input_dict["images_clip"],
                    input_dict["images_imis"],
                    input_dict["input_ids_seg"],
                    input_dict["attention_masks_seg"],
                    input_dict["input_ids"],
                    input_dict["attention_masks"]
                )
            else:
                output_ids = model_engine.generate(
                    input_ids,
                    images=input_dict['images_clip'],
                    region_masks=None,
                    valid_region_masks_bool=None,
                    attention_mask=attention_mask,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "image_path": input_dict['image_paths'][0],
                                   "prompt": input_dict['questions_list'][0][0],
                                   "gt": input_dict['answers_list'][0][0],
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "answer_type": args.answer_type,
                                   "data_type": input_dict['data_type'][0] if 'hallucination' in args.benchmark_name else 'omniMedVQA',
                                   "candidate_list": candidate_list,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
        idx += 1
    ans_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
