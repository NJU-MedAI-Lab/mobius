#!/bin/bash
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="mobius-stage3-proj-only"
exp_dir="/workspace/EviMed/Mobius_runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=65000 train_ds_mobius.py \
  --version="/workspace/EviMed/checkpoints/mobius-stage2-Long+seg-1152" \
  --pretrain_sam="/workspace/EviMed/checkpoints/mobius-stage2-Long+seg-1152" \
  --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
  --data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage1_Long-cleaned.json' \
  --data_path_seg='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Seg_for_proj-cleaned.json' \
  --val_data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage1_Long-cleaned.json' \
  --val_data_path_seg='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Seg_for_proj-cleaned.json' \
  --image_folder='/workspace/EviMed/IMIS_data/' \
  --imis_sam_checkpoint="/workspace/EviMed/huggingface/IMISNet-B.pth" \
  --log_base_dir="/workspace/EviMed/Mobius_runs" \
  --vision_pretrained="/workspace/EviMed/huggingface/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=2 \
  --batch_size=64 \
  --workers=16 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 3 \
  --imis_image_size 1024 \
  --out_dim 768 \
  --sam_ce_loss_weight 1.0 \
  --lora_r 0 \
  --sft_modules_sam "imis_to_vlm_proj_fcs" \
  --lr 0.0001 \
  --save_steps 20 \
  --eval_steps 20 \
  --proj_only \
  2>&1 | tee -a $exp_dir/$time.log
