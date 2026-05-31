#!/bin/bash
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="mobius-stage2-Long+seg"
exp_dir="/workspace/EviMed/Mobius_runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=65000 train_ds_mobius.py \
  --version="/workspace/EviMed/checkpoints/mobius-stage1-Long+short+seg+other-VQA-1816" \
  --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
  --data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage2_Long+Seg-cleaned.json' \
  --val_data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage2_Long+Seg-cleaned.json' \
  --image_folder='/workspace/EviMed/IMIS_data/' \
  --imis_sam_checkpoint="/workspace/EviMed/huggingface/IMISNet-B.pth" \
  --log_base_dir="/workspace/EviMed/IMIS_runs/" \
  --vision_pretrained="/workspace/EviMed/huggingface/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=10 \
  --batch_size=64 \
  --workers=16 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 3 \
  --imis_image_size 1024 \
  --out_dim 768 \
  --vlm_mask_loss_weight 1.0 \
  --sft_modules_vlm "text_hidden_fcs" \
  --sft_modules_sam "mask_decoder" \
  --lr 0.00015 \
  --save_steps 100 \
  --eval_steps 100 \
  --train_mask_decoder \
  --sam_only \
  2>&1 | tee -a $exp_dir/$time.log
