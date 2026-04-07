import collections
import difflib

from tomlkit import datetime
from model.eval.glossary import normalize_word
from model.eval.evaluate_metrics import calculate_exactmatch, calculate_f1score, bleu, calculate_appearance_with_normalization
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, ADD_OTHERS_TOKENS)
from datasets import (DataCollatorForSupervisedDataset_stage3pre, LazySupervisedDataset_stage3pre,
                      LazySupervisedDataset_stage1, DataCollatorForSupervisedDataset_stage1,
                      LazySupervisedDataset_stage2, DataCollatorForSupervisedDataset_stage2,
                      MedicalDataset_omni, DataCollatorForOmniMedVQADataset,
                      MedicalDataset_hallu, DataCollatorForObjectHallucinationDataset)
from model.mobius import conversation as conversation_lib
from model.Mobius import MobiusForCausalLM
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model
import transformers
from transformers import AutoConfig, get_cosine_schedule_with_warmup
import tqdm
import torch
import numpy as np
import deepspeed
import random
import math
import types
from functools import partial
import time
import sys
import shutil
import argparse
import os
import warnings
import torch.distributed as dist
from torch.distributed import all_gather_object
import json
warnings.filterwarnings("ignore")

local_rank = None


def parse_args(args):
    parser = argparse.ArgumentParser(description="Mobius Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    # ----------------- Model setting -------------------
    parser.add_argument(
        "--version", default="/data/huggingface/llava-v1.5-7b/"
    )
    parser.add_argument("--vis_save_path", default="/vis_output", type=str)
    parser.add_argument("--pretrain_mm_mlp_adapter",
                        default=None, type=str)
    parser.add_argument("--pretrain_sam", default=None, type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--sam_img_size", default=256,
                        type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
    parser.add_argument(
        "--vision_tower", default='/mydata/huggingface/clip-vit-large-patch14-336', type=str
    )
    parser.add_argument("--vision_pretrained",
                        default="/mydata/huggingface/sam-med2d_b.pth", type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--sft_modules_vlm", default="",
                        type=str, help='set the supervised fintuning modules')
    parser.add_argument("--sft_modules_sam", default="",
                        type=str, help='set the supervised fintuning modules')
    parser.add_argument("--vlm_only", action="store_true", default=False)
    parser.add_argument("--sam_only", action="store_true", default=False)
    parser.add_argument("--proj_only", action="store_true", default=False)

    # ------------------ Lora setting -------------------
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules",
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str)

    # ---------------- Dataset setting -----------------
    parser.add_argument('--image_folder', type=str, default='/mydata/IMIS_data/',
                        help='Path to the folder containing images.')
    parser.add_argument("--sub_image_folder", type=str,
                        default="/mydata/IMIS_data/")
    parser.add_argument('--image_aspect_ratio', type=str,
                        default='pad', help='How to handle image aspect ratio.')
    parser.add_argument('--is_multimodal', type=bool,
                        default=True, help='Whether to use multimodal data.')
    parser.add_argument('--data_path', type=str, default="/mydata/IMIS_data/Stage2_Long+Short+otherVQA.json",
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--data_path_seg', type=str, default="/mydata/IMIS_data/Stage2_Long+Short+otherVQA.json",
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--val_data_path', type=str, default="/mydata/IMIS_data/Stage2_Long+Short+otherVQA.json",
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--val_data_path_seg', type=str, default="/mydata/IMIS_data/Stage2_Long+Short+otherVQA.json",
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--val_hallucination_path', type=str, default=None,
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--val_omni_path', type=str, default=None,
                        help='Path to the JSON file containing the data.')
    parser.add_argument('--val_image_folder', type=str, default='/mydata/IMIS_data/',
                        help='Path to the folder containing validation images.')

    # ---------------- Training setting -----------------
    parser.add_argument(
        "--log_base_dir", default="/mydata/IMIS_runs/runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument(
        "--batch_size", default=16, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=2,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--vlm_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--sam_ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--vlm_mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--sam_mask_loss_weight", default=1.0, type=float)
    # parser.add_argument("--focal_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--save_steps", default=400, type=int)
    parser.add_argument("--eval_steps", default=400, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing",
                        action="store_true", default=True)
    parser.add_argument("--train_mask_decoder",
                        action="store_true", default=False)
    parser.add_argument("--train_prompt_encoder",
                        action="store_true", default=False)
    parser.add_argument("--use_mm_start_end",
                        action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--early_stop_vlm_step", default=10000, type=int)
    parser.add_argument("--early_stop_sam_step", default=10000, type=int)

    # IMIS
    parser.add_argument('--imis_work_dir', type=str, default='work_dir')
    parser.add_argument('--imis_task_name', type=str, default='ft-IMISNet')
    # load data
    parser.add_argument("--imis_data_dir", type=str, default='dataset/BTCV')
    parser.add_argument('--imis_image_size', type=int, default=1024)
    parser.add_argument('--imis_test_mode', type=bool, default=False)
    parser.add_argument('--imis_batch_size', type=int, default=10)
    # load model
    parser.add_argument('--imis_model_type', type=str, default='vit_b')
    parser.add_argument('--imis_sam_checkpoint', type=str,
                        default="/data/huggingface/IMISNet-B.pth")
    parser.add_argument('--imis_pretrain_path', type=str,
                        default="/data/huggingface/IMISNet-B.pth")
    parser.add_argument('--imis_resume', action='store_true', default=True)
    parser.add_argument('--imis_device', type=str, default='cuda')
    parser.add_argument('--imis_mask_num', type=int, default=2)
    parser.add_argument('--imis_inter_num', type=int, default=4)
    # train
    parser.add_argument('--imis_num_epochs', type=int, default=20)
    parser.add_argument('--imis_lr_scheduler', type=str, default=None)
    parser.add_argument('--imis_step_size', type=list, default=[7, 12])
    parser.add_argument('--imis_gamma', type=float, default=0.5)
    parser.add_argument('--imis_lr', type=float, default=1e-4)
    parser.add_argument('--imis_weight_decay', type=float, default=1e-5)
    parser.add_argument('--imis_port', type=int, default=12305)
    parser.add_argument('--imis_gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--imis_multi_gpu', action='store_true', default=False)
    parser.add_argument('--imis_dist', dest='dist', type=bool,
                        default=False, help='distributed training or not')
    parser.add_argument('--imis_num_workers', type=int, default=1)

    return parser.parse_args(args)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_trainable_parts(model, mode, args):
    for n, p in model.named_parameters():
        p.requires_grad = False
    if mode == "vlm_sam":
        if args.early_stop_vlm_step < model.global_steps:
            if model.global_steps == args.early_stop_vlm_step + 1:
                print("Early stop vlm training at step ", model.global_steps)
            return

        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        if args.sft_modules_vlm != "":
            sft_modules = args.sft_modules_vlm.split(",")
            for n, p in model.named_parameters():
                if any(
                    [
                        x in n
                        for x in sft_modules
                    ]
                ):
                    p.requires_grad = True
    elif mode == "sam_vlm":
        if args.early_stop_sam_step < model.global_steps:
            if model.global_steps == args.early_stop_sam_step + 1:
                print("Early stop sam training at step ", model.global_steps)
            return
        if args.proj_only:
            for n, p in model.named_parameters():
                if "imis_to_vlm_proj_fcs" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            if args.sft_modules_sam != "":
                sft_modules = args.sft_modules_sam.split(",")
                for n, p in model.named_parameters():
                    if any(
                        [
                            x in n
                            for x in sft_modules
                        ]
                    ):
                        p.requires_grad = True


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


def main(args):
    global local_rank
    args = parse_args(args)
    local_rank = args.local_rank
    rank0_print("local rank:", local_rank)
    set_seed(42)
    if args.local_rank == 0:
        test_randomness()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0 and not args.eval_only:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    for i in range(1, 257):
        ADD_OTHERS_TOKENS.append("<gen_" + str(i) + ">")
    for token_name in ADD_OTHERS_TOKENS:
        tokenizer.add_tokens(token_name, special_tokens=True)
    args.seg_token_idx = tokenizer(
        "<SEG>", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = vars(args)
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    # load LLM and tower weights
    model = MobiusForCausalLM.from_pretrained(
        args.version,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # load tower and projector weights
    model.config.pretrain_mm_mlp_adapter = args.pretrain_mm_mlp_adapter
    model.get_model().initialize_vision_modules(model.get_model().config)
    if not args.eval_only:
        # load sam and text_hidden_fcs weights
        model.get_model().initialize_imis_modules(model.get_model().config, vars(args))

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0 and not args.sam_only:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                # "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        if args.local_rank == 0:
            rank0_print('lora_target_modules', lora_target_modules)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for n, p in model.named_parameters():
            p.requires_grad = False

    model.resize_token_embeddings(len(tokenizer))

    # vlm trainable parameters without lora
    if args.sft_modules_vlm != "":
        sft_modules = args.sft_modules_vlm.split(",")
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in sft_modules
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    # sam trainable parameters without lora
    if args.sft_modules_sam != "":
        sft_modules = args.sft_modules_sam.split(",")
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in sft_modules
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    data_args = {"image_folder": args.image_folder,
                 "val_image_folder": args.val_image_folder,
                 "sub_image_folder": args.sub_image_folder,
                 "image_aspect_ratio": args.image_aspect_ratio,
                 "is_multimodal": args.is_multimodal,
                 "mm_use_im_start_end": args.use_mm_start_end,
                 "image_processor": vision_tower.image_processor
                 }
    data_args = types.SimpleNamespace(**data_args)
    if args.vlm_only:
        train_dataset = LazySupervisedDataset_stage1(
            args.data_path, tokenizer, data_args, args.imis_image_size, mode='train')
    elif args.sam_only:
        train_dataset = LazySupervisedDataset_stage2(
            args.data_path, tokenizer, data_args, args.imis_image_size, mode='train')
    else:
        train_dataset = LazySupervisedDataset_stage3pre(
            [args.data_path_seg, args.data_path], tokenizer, data_args, args.imis_image_size, args.proj_only, mode='train')

    args.steps_per_epoch = math.ceil(math.ceil(len(
        train_dataset) / (args.batch_size * torch.cuda.device_count())) / args.grad_accumulation_steps)

    if args.no_eval == False:
        if args.vlm_only:
            val_dataset = LazySupervisedDataset_stage1(
                args.val_data_path, tokenizer, data_args, args.imis_image_size, mode='val')
            if args.val_hallucination_path is not None:
                val_hallu_dataset = MedicalDataset_hallu(
                    args.val_hallucination_path, tokenizer, data_args)
            if args.val_omni_path is not None:
                val_omni_dataset = MedicalDataset_omni(
                    args.val_omni_path, tokenizer, data_args)
        elif args.sam_only:
            val_dataset = LazySupervisedDataset_stage2(
                args.val_data_path, tokenizer, data_args, args.imis_image_size, mode='val')
        elif args.proj_only:
            val_dataset = LazySupervisedDataset_stage3pre(
                [args.val_data_path_seg, args.val_data_path], tokenizer, data_args, args.imis_image_size, args.proj_only, mode='val')
        else:
            val_dataset = LazySupervisedDataset_stage3pre(
                [args.val_data_path_seg, args.val_data_path], tokenizer, data_args, args.imis_image_size, args.proj_only, mode='val')
            if args.val_hallucination_path is not None:
                val_hallu_dataset = MedicalDataset_hallu(
                    args.val_hallucination_path, tokenizer, data_args)
            if args.val_omni_path is not None:
                val_omni_dataset = MedicalDataset_omni(
                    args.val_omni_path, tokenizer, data_args)

        rank0_print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples. steps in one epoch: {args.steps_per_epoch}"
        )
    else:
        val_dataset = None
        rank0_print(
            f"Training with {len(train_dataset)} examples. steps in one epoch: {args.steps_per_epoch}")

    total_num_step = args.epochs*args.steps_per_epoch
    warmup_step = int(args.steps_per_epoch * args.warmup_ratio)

    if args.vlm_only or args.sam_only or args.proj_only:
        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": args.lr,
                    "weight_decay": 0.0,
                    "betas": (args.beta1, args.beta2),
                },
            },
            "gradient_clipping": 1.0,
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "total_num_steps": total_num_step,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": args.lr,
                    "warmup_num_steps": warmup_step,
                    "warmup_type": "linear",
                },
            },
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            },
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }
    else:
        vlm_params = []
        imis_params = []
        for n, p in model.named_parameters():
            if "visual_model" in n or "imis_to_vlm_proj_fcs" in n:
                imis_params.append(p)
            else:
                vlm_params.append(p)

        # param groups
        param_groups = [
            {"params": vlm_params, "lr": args.lr, "weight_decay": 0.01,
             "betas": (args.beta1, args.beta2)},
            {"params": imis_params, "lr": args.imis_lr, "weight_decay": 0.0,
             "betas": (args.beta1, args.beta2)},
        ]

        optimizer = torch.optim.AdamW(
            param_groups, betas=(args.beta1, args.beta2))

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_num_step
        )

        ds_config = {
            "train_micro_batch_size_per_gpu": args.batch_size,
            "gradient_accumulation_steps": args.grad_accumulation_steps,
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": args.precision == "fp16",
            },
            "bf16": {
                "enabled": args.precision == "bf16",
            },
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
        }

    if args.vlm_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            # training_data=None,
            collate_fn=partial(
                DataCollatorForSupervisedDataset_stage1,
            ),
            config=ds_config,
        )
    elif args.sam_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            # training_data=None,
            collate_fn=partial(
                DataCollatorForSupervisedDataset_stage2,
            ),
            config=ds_config,
        )
    else:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=None,
            training_data=train_dataset,
            # training_data=None,
            collate_fn=partial(
                DataCollatorForSupervisedDataset_stage3pre,
            ),
            config=ds_config,
        )

    rank0_print(f"deepspeed.initialize done!!!!!!!!!!")

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        rank0_print('The resume model global_steps is ',
                    model_engine.global_steps)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        rank0_print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        if args.vlm_only:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    DataCollatorForSupervisedDataset_stage1,
                    inference=True,
                ),
            )
            if args.val_hallucination_path is not None:
                val_sampler_1 = torch.utils.data.distributed.DistributedSampler(
                    val_hallu_dataset, shuffle=False, drop_last=False
                )
                val_hallu_loader = torch.utils.data.DataLoader(
                    val_hallu_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=False,
                    sampler=val_sampler_1,
                    collate_fn=partial(
                        DataCollatorForObjectHallucinationDataset,
                        inference=True,
                    ),
                )
            if args.val_omni_path is not None:
                val_sampler_2 = torch.utils.data.distributed.DistributedSampler(
                    val_omni_dataset, shuffle=False, drop_last=False
                )
                val_omni_loader = torch.utils.data.DataLoader(
                    val_omni_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=False,
                    sampler=val_sampler_2,
                    collate_fn=partial(
                        DataCollatorForOmniMedVQADataset,
                        inference=True,
                    ),
                )
        elif args.sam_only:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    DataCollatorForSupervisedDataset_stage2,
                    inference=True,
                ),
            )
        elif args.proj_only:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    DataCollatorForSupervisedDataset_stage3pre,
                    inference=True,
                ),
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    DataCollatorForSupervisedDataset_stage3pre,
                    inference=True,
                ),
            )
            if args.val_hallucination_path is not None:
                val_sampler_1 = torch.utils.data.distributed.DistributedSampler(
                    val_hallu_dataset, shuffle=False, drop_last=False
                )
                val_hallu_loader = torch.utils.data.DataLoader(
                    val_hallu_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=False,
                    sampler=val_sampler_1,
                    collate_fn=partial(
                        DataCollatorForObjectHallucinationDataset,
                        inference=True,
                    ),
                )
            if args.val_omni_path is not None:
                val_sampler_2 = torch.utils.data.distributed.DistributedSampler(
                    val_omni_dataset, shuffle=False, drop_last=False
                )
                val_omni_loader = torch.utils.data.DataLoader(
                    val_omni_dataset,
                    batch_size=args.val_batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    pin_memory=False,
                    sampler=val_sampler_2,
                    collate_fn=partial(
                        DataCollatorForOmniMedVQADataset,
                        inference=True,
                    ),
                )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        if args.vlm_only:
            validate_vlm(val_loader, model_engine,
                         tokenizer, writer, 0, args)
            if args.val_omni_path is not None:
                validate(val_omni_loader, model_engine, writer, 0, args)
            if args.val_hallucination_path is not None:
                validate_vlm(val_hallu_loader, model_engine,
                             tokenizer, writer, 0, args)
        else:
            giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter, best_score, cur_ciou = train(
            train_loader,
            val_loader,
            val_hallu_loader if args.val_hallucination_path is not None else None,
            val_omni_loader if args.val_omni_path is not None else None,
            model_engine,
            tokenizer,
            optimizer,
            epoch,
            scheduler,
            writer,
            train_iter,
            best_score,
            cur_ciou,
            args,
        )

        if args.no_eval == False:
            if not args.vlm_only and not args.proj_only:
                giou, ciou = validate(
                    val_loader, model_engine, model_engine.global_steps, writer, args)
                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou

                if is_best:
                    save_dir = os.path.join(args.log_dir, "ckpt_model_best")
                    if args.local_rank == 0:
                        torch.save(
                            {"epoch": epoch},
                            os.path.join(
                                args.log_dir,
                                "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                                    best_score, cur_ciou
                                ),
                            ),
                        )
                        if os.path.exists(save_dir):
                            shutil.rmtree(save_dir)
                    torch.distributed.barrier()
                    model_engine.save_checkpoint(save_dir)

        save_dir = os.path.join(args.log_dir, f"ckpt_model_epoch_{epoch+1}")
        if args.local_rank == 0:
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        torch.distributed.barrier()
        model_engine.save_checkpoint(save_dir)

        torch.cuda.empty_cache()


def train(
    train_loader,
    val_loader,
    val_hallu_loader,
    val_omni_loader,
    model,
    tokenizer,
    optimizer,
    epoch,
    scheduler,
    writer,
    train_iter,
    best_score,
    cur_ciou,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    losses = AverageMeter("Loss", ":.4f")
    imis_total_losses = AverageMeter("IMISTotalLoss", ":.4f")
    vlm_total_losses = AverageMeter("VLMTotalLoss", ":.4f")
    imis_ce_losses = AverageMeter("IMISCeLoss", ":.4f")
    imis_mask_losses = AverageMeter("IMISMaskLoss", ":.4f")
    vlm_ce_losses = AverageMeter("VLMCeLoss", ":.4f")
    vlm_mask_losses = AverageMeter("VLMMaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            model.global_steps,
            batch_time,
            data_time,
            losses,
            imis_total_losses,
            imis_ce_losses,
            imis_mask_losses,
            vlm_total_losses,
            vlm_ce_losses,
            vlm_mask_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()

    first_step_in_epoch = model.global_steps % args.steps_per_epoch

    if first_step_in_epoch > 0:
        rank0_print('skipping first {} steps, global step is {}'.format(
            first_step_in_epoch, model.global_steps))

        for local_step in tqdm.tqdm(range(0, first_step_in_epoch)):
            for i in range(args.grad_accumulation_steps):
                try:
                    input_dict = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    input_dict = next(train_iter)

    for local_step in range(first_step_in_epoch, args.steps_per_epoch):
        optimizer.zero_grad()
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
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

            input_dict["vlm_only"] = args.vlm_only
            input_dict["sam_only"] = args.sam_only
            input_dict["proj_only"] = args.proj_only

            input_dict["sft_modules_vlm"] = args.sft_modules_vlm
            input_dict["sft_modules_sam"] = args.sft_modules_sam

            if args.vlm_only:
                # ----------- only train vlm ------------
                vlm_output_dict = model(mode="vlm_sam", **input_dict)
                loss = vlm_output_dict["loss"]
                losses.update(
                    loss.item(), input_dict["images_imis"].size(0))
                model.backward(loss)
                model.step()
            elif args.sam_only:
                # ----------- only train sam ------------
                vlm_output_dict = model(mode="vlm_sam", **input_dict)
                loss = vlm_output_dict["loss"]
                losses.update(
                    loss.item(), input_dict["images_imis"].size(0))
                model.backward(loss)
                model.step()
            elif args.proj_only:
                # ----------- only train imis_to_vlm_proj ------------
                vlm_output_dict = model(mode="sam_vlm", **input_dict)
                loss = vlm_output_dict["loss"]
                losses.update(
                    loss.item(), input_dict["images_imis"].size(0))
                model.backward(loss)
                model.step()

            if args.vlm_only:
                vlm_total_loss = vlm_output_dict["loss"]
                vlm_ce_loss = vlm_output_dict["ce_loss"]
                vlm_mask_loss = vlm_output_dict["mask_loss"]

                vlm_total_losses.update(vlm_total_loss.item(),
                                        input_dict["images_imis"].size(0))
                vlm_ce_losses.update(vlm_ce_loss.item(),
                                     input_dict["images_imis"].size(0))

                del vlm_output_dict, vlm_total_loss, vlm_ce_loss, vlm_mask_loss

            elif args.sam_only:
                vlm_total_loss = vlm_output_dict["loss"]
                vlm_ce_loss = vlm_output_dict["ce_loss"]
                vlm_mask_loss = vlm_output_dict["mask_loss"]

                vlm_total_losses.update(vlm_total_loss.item(),
                                        input_dict["images_imis"].size(0))
                vlm_mask_losses.update(vlm_mask_loss.item(),
                                       input_dict["images_imis"].size(0))

                del vlm_output_dict, vlm_total_loss, vlm_ce_loss, vlm_mask_loss

            elif args.proj_only:
                vlm_total_loss = vlm_output_dict["loss"]
                vlm_ce_loss = vlm_output_dict["ce_loss"]
                vlm_mask_loss = vlm_output_dict["mask_loss"]

                vlm_total_losses.update(vlm_total_loss.item(),
                                        input_dict["images_imis"].size(0))
                vlm_ce_losses.update(vlm_ce_loss.item(),
                                     input_dict["images_imis"].size(0))

                del vlm_output_dict, vlm_total_loss, vlm_ce_loss, vlm_mask_loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if local_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                imis_total_losses.all_reduce()
                imis_ce_losses.all_reduce()
                imis_mask_losses.all_reduce()
                vlm_total_losses.all_reduce()
                vlm_ce_losses.all_reduce()
                vlm_mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(local_step + 1)
                # writer.add_scalar("train/loss", losses.avg, model.global_steps)

                if args.vlm_only:
                    writer.add_scalar(
                        "train/loss", losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_total_loss", vlm_total_losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_ce_loss", vlm_ce_losses.avg, model.global_steps)
                elif args.sam_only:
                    writer.add_scalar(
                        "train/loss", losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_total_loss", vlm_total_losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_mask_loss", vlm_mask_losses.avg, model.global_steps)

                elif args.proj_only:
                    writer.add_scalar(
                        "train/loss", losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_total_loss", vlm_total_losses.avg, model.global_steps)
                    writer.add_scalar(
                        "train/vlm_ce_loss", vlm_ce_losses.avg, model.global_steps)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            imis_total_losses.reset()
            imis_ce_losses.reset()
            imis_mask_losses.reset()
            vlm_total_losses.reset()
            vlm_ce_losses.reset()
            vlm_mask_losses.reset()

        if model.global_steps != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], model.global_steps)

        if args.no_eval == False and (local_step+1) % args.eval_steps == 0:
            if args.vlm_only:
                # validate evimed val
                acc = validate_vlm(val_loader, model,
                                   tokenizer, writer, model.global_steps, args, type="open-vqa")
                is_best = acc > best_score
                best_score = max(acc, best_score)
                # validate hallucination
                if args.val_hallucination_path is not None:
                    _ = validate_vlm(val_hallu_loader, model,
                                     tokenizer, writer, model.global_steps, args)
                # validate omnimedvqa
                if args.val_omni_path is not None:
                    validate(val_omni_loader, model,
                             model.global_steps, writer, args, vlm_only=True)
                    _ = validate_vlm(val_omni_loader, model,
                                     tokenizer, writer, model.global_steps, args, type="OmniMedVQA")
            elif args.sam_only:
                giou, ciou = validate(
                    val_loader, model, model.global_steps, writer, args)
                is_best = giou > best_score
                best_score = max(giou, best_score)
                cur_ciou = ciou if is_best else cur_ciou

            elif args.proj_only:
                acc = validate_vlm(val_loader, model,
                                   tokenizer, writer, model.global_steps, args, type="open-vqa")
                is_best = acc > best_score
                best_score = max(acc, best_score)

            if is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model_best")
                if args.local_rank == 0:
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model.save_checkpoint(save_dir)

        if local_step != 0 and (local_step+1) % args.save_steps == 0:  # 每 100 step 保存一次
            save_dir = os.path.join(
                args.log_dir, f"ckpt_model")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model.save_checkpoint(save_dir)

        if local_step != 0 and (local_step+1) % 200 == 0:  # 每 100 step 保存一次
            save_dir = os.path.join(
                args.log_dir, f"ckpt_model_{model.global_steps}")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model.save_checkpoint(save_dir)
        torch.cuda.empty_cache()

    return train_iter, best_score, cur_ciou


def calculate_iou(prediction_mask, ground_truth_mask):
    # to boolen
    prediction_mask = prediction_mask.bool()
    ground_truth_mask = ground_truth_mask.bool()

    # cal i and u
    intersection = torch.logical_and(prediction_mask, ground_truth_mask)
    union = torch.logical_or(prediction_mask, ground_truth_mask)

    intersection_pixels = torch.sum(intersection)
    union_pixels = torch.sum(union)

    if union_pixels == 0:
        iou = 0
    else:
        iou = intersection_pixels.float() / union_pixels.float()

    return iou.item()


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = 0
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


def validate_vlm(val_loader, model_engine, tokenizer, writer, epoch, args, type="Object Hallucination"):
    model_engine.eval()

    scores = collections.defaultdict(list)
    exact_scores = collections.defaultdict(list)
    scores_modal_dict = {}
    scores_modal_dict_exact = {}
    pbar = tqdm.tqdm(val_loader)
    for input_dict in pbar:
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)

        input_dict["tokenizer"] = tokenizer
        answer_type = input_dict["answer_type"][0]

        with torch.no_grad():
            if args.proj_only:
                if args.precision == "fp16":
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                    input_dict["images_imis"] = input_dict["images_imis"].half()
                elif args.precision == "bf16":
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16(
                    )
                    input_dict["images_imis"] = input_dict["images_imis"].bfloat16(
                    )
                else:
                    input_dict["images_clip"] = input_dict["images_clip"].float()
                    input_dict["images_imis"] = input_dict["images_imis"].float()
                output = model_engine.module.evaluate_sam_vlm(**input_dict)
                gt_value = normalize_word(
                    input_dict['answers_list'][0][0].lower())
                norm_output = normalize_word(output.lower())
            else:
                if args.precision == "fp16":
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images_clip"] = input_dict["images_clip"].float()
                output = model_engine.module.evaluate_vlm(**input_dict)
                gt_value = normalize_word(
                    input_dict['answers_list'][0][0].lower())
                norm_output = normalize_word(output.lower())

        if answer_type.lower() in ["open", "other", "number"]:
            scores['hit'].append(calculate_exactmatch(norm_output, gt_value))
            exact_scores['hit'].append(
                calculate_exactmatch(norm_output, gt_value))
        elif answer_type.lower() in ["yes/no", "closed"]:
            answer_list = input_dict['candidate_list'][0]
            norm_answer_list = []
            for answer in answer_list:
                norm_answer_list.append(normalize_word(answer.lower()))
            idx, sim_str = find_most_similar_index(
                norm_answer_list, norm_output)

            if "Hallucination" in type:
                type_ = input_dict['data_type'][0]
                if type_ not in scores_modal_dict:
                    scores_modal_dict[type_] = collections.defaultdict(list)
                    scores_modal_dict_exact[type_] = collections.defaultdict(
                        list)

                if 'yes' in norm_output or 'no' in norm_output:
                    if gt_value == norm_output:
                        scores_modal_dict_exact[type_]['hit'].append(1)
                    else:
                        scores_modal_dict_exact[type_]['hit'].append(0)
                else:
                    scores_modal_dict_exact[type_]['hit'].append(0)

                if gt_value == norm_answer_list[idx]:
                    scores_modal_dict[type_]['hit'].append(1)
                else:
                    scores_modal_dict[type_]['hit'].append(0)

            else:
                if gt_value == norm_answer_list[idx]:
                    scores['hit'].append(1)
                else:
                    scores['hit'].append(0)

                if gt_value == norm_output:
                    exact_scores['hit'].append(1)
                else:
                    exact_scores['hit'].append(0)

    torch.distributed.barrier()
    gathered_results = [None for _ in range(
        torch.distributed.get_world_size())]
    gathered_exact_results = [None for _ in range(
        torch.distributed.get_world_size())]
    if "Hallucination" in type:
        all_gather_object(gathered_results, scores_modal_dict)
        all_gather_object(gathered_exact_results, scores_modal_dict_exact)

        mean_lst = []
        mean_lst_exact = []
        gathered_modal_scores = {}
        for res in gathered_results:
            for key in res:
                if key not in gathered_modal_scores:
                    gathered_modal_scores[key] = collections.defaultdict(list)
                gathered_modal_scores[key]['hit'].extend(res[key]['hit'])
        for k in gathered_modal_scores.keys():
            acc = sum(gathered_modal_scores[k]['hit']) / len(
                gathered_modal_scores[k]['hit']) if len(gathered_modal_scores[k]['hit']) != 0 else 0.0
            mean_lst.extend(gathered_modal_scores[k]['hit'])
            if args.local_rank == 0 and not args.eval_only:
                writer.add_scalar(f"val/{type}_{k}_closed_acc", acc, epoch)
                print(f"{type} {k} closed acc: {acc*100:.4f}")
        gathered_modal_exact_scores = {}
        for res in gathered_exact_results:
            for key in res:
                if key not in gathered_modal_exact_scores:
                    gathered_modal_exact_scores[key] = collections.defaultdict(
                        list)
                gathered_modal_exact_scores[key]['hit'].extend(res[key]['hit'])
        for k in gathered_modal_exact_scores.keys():
            exact_acc = sum(gathered_modal_exact_scores[k]['hit']) / len(
                gathered_modal_exact_scores[k]['hit']) if len(gathered_modal_exact_scores[k]['hit']) != 0 else 0.0
            mean_lst_exact.extend(gathered_modal_exact_scores[k]['hit'])
            if args.local_rank == 0 and not args.eval_only:
                writer.add_scalar(
                    f"val/{type}_{k}_closed_exact_acc", exact_acc, epoch)
                print(f"{type} {k} closed exact acc: {exact_acc*100:.4f}")
        acc = sum(mean_lst) / len(mean_lst) if len(mean_lst) != 0 else 0.0
        exact_acc = sum(mean_lst_exact) / \
            len(mean_lst_exact) if len(mean_lst_exact) != 0 else 0.0

        del gathered_modal_scores, scores_modal_dict, mean_lst, gathered_modal_exact_scores, scores_modal_dict_exact, mean_lst_exact
    else:
        all_gather_object(gathered_results, scores)
        all_gather_object(gathered_exact_results, exact_scores)

        gathered_scores = collections.defaultdict(list)
        for res in gathered_results:
            gathered_scores['hit'].extend(res['hit'])
        acc = sum(gathered_scores['hit']) / len(
            gathered_scores['hit']) if len(gathered_scores['hit']) != 0 else 0.0
        gathered_exact_scores = collections.defaultdict(list)
        for res in gathered_exact_results:
            gathered_exact_scores['hit'].extend(res['hit'])
        exact_acc = sum(gathered_exact_scores['hit']) / len(
            gathered_exact_scores['hit']) if len(gathered_exact_scores['hit']) != 0 else 0.0

        del gathered_scores, scores, gathered_exact_scores, exact_scores
    if args.local_rank == 0 and not args.eval_only:
        if answer_type.lower() in ["open", "other", "number"]:
            writer.add_scalar("val/open_acc", acc, epoch)
            print(f"VLM open acc: {acc*100:.4f}")
        elif answer_type.lower() in ["yes/no", "closed"]:
            writer.add_scalar(f"val/{type}_closed_mean_acc", acc, epoch)
            writer.add_scalar(
                f"val/{type}_closed_mean_exact_acc", exact_acc, epoch)
            print(f"{type} closed mean exact acc: {exact_acc*100:.4f}")
            print(f"{type} closed mean acc: {acc*100:.4f}")
    torch.cuda.empty_cache()
    model_engine.train()

    return acc


def validate(val_loader, model_engine, epoch, writer, args, vlm_only=False):
    if vlm_only:
        total_loss = AverageMeter("TotalLoss", ":.4f", Summary.SUM)
        model_engine.eval()
        pbar = tqdm.tqdm(val_loader)
        for input_dict in pbar:
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images_clip"] = input_dict["images_clip"].float()

            input_dict["vlm_only"] = True

            with torch.no_grad():
                outputs = model_engine(mode="vlm_sam", **input_dict)

            total_loss.update(outputs["loss"].item(),
                              input_dict["images_clip"].size(0))

        total_loss.all_reduce()
        val_loss = total_loss.avg
        if args.local_rank == 0 and not args.eval_only:
            writer.add_scalar("val/omni_total_loss", val_loss, epoch)
            print(f"OmniMedVQA total loss: {val_loss:.4f}")

        del outputs, total_loss
        torch.cuda.empty_cache()
        model_engine.train()
        return val_loss

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)

    scores = collections.defaultdict(list)

    model_engine.eval()

    pbar = tqdm.tqdm(val_loader)
    for input_dict in pbar:
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

        if args.sam_only:
            input_dict["sam_only"] = True
        else:
            input_dict["sam_only"] = False

        answer_type = input_dict["answer_type"][0]

        with torch.no_grad():
            outputs, pred_masks, gt_masks = model_engine.module.evaluate(
                **input_dict)

        gt_value = normalize_word(
            input_dict['answers_list'][0][0].lower())
        norm_output = normalize_word(outputs.lower())

        if answer_type.lower() in ["open", "other", "number"]:
            scores['hit'].append(calculate_exactmatch(norm_output, gt_value))

        output_list = (torch.sigmoid(pred_masks[0]) > 0.1).int()
        # assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(gt_masks, output_list):
            mask_i = mask_i.unsqueeze(0)
            output_i = output_i.unsqueeze(0)
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
            iou = calculate_iou(output_i, mask_i)
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / len(gt_masks)
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=len(gt_masks))

        iou_meter.update(iou)
        dice_meter.update(2*iou/(1+iou))

        new_description = f'iou={acc_iou}.'
        pbar.set_description(new_description)

    torch.distributed.barrier()
    gathered_results = [None for _ in range(
        torch.distributed.get_world_size())]
    all_gather_object(gathered_results, scores)

    gathered_scores = collections.defaultdict(list)
    for res in gathered_results:
        gathered_scores['hit'].extend(res['hit'])
    acc = sum(gathered_scores['hit']) / len(
        gathered_scores['hit']) if len(gathered_scores['hit']) != 0 else 0.0
    if args.local_rank == 0 and not args.eval_only:
        print(f"VLM Evaluating {len(gathered_scores['hit'])} samples")
        if answer_type.lower() in ["open", "other", "number"]:
            writer.add_scalar("val/open_acc", acc, epoch)
            print(f"VLM open acc: {acc*100:.4f}")

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    iou_meter.all_reduce()
    dice_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    miou = iou_meter.avg
    mDice = dice_meter.avg

    if args.local_rank == 0 and not args.eval_only:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/dice", mDice, epoch)
        print("giou: {:.6f}, ciou: {:.6f}".format(giou, ciou))
        print("miou: {:.6f}, mDice: {:.6f}".format(miou, mDice))

    del pred_masks, gt_masks, outputs
    del gathered_scores, scores
    torch.cuda.empty_cache()
    model_engine.train()
    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
