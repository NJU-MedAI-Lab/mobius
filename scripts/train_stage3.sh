#!/bin/bash
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="mobius-stage3-loop-alllong-bs48-3e-5-5e-5-trainfcs-gradaccu5-vlmearlystop60"
exp_dir="/workspace/EviMed/Mobius_runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:2,3 --master_port=65000 train_ds_mobius_stage3.py \
  --version="/workspace/EviMed/checkpoints/mobius-stage3-pre-296" \
  --pretrain_sam="/workspace/EviMed/checkpoints/mobius-stage3-pre-296" \
  --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
  --data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Long-Train.json' \
  --data_path_seg='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Long_Seg-Train.json' \
  --val_data_path='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Long-Train.json' \
  --val_data_path_seg='/workspace/EviMed/IMIS_data_jsons_cleaned/Stage3_Long_Seg-Train.json' \
  --val_hallucination_path="/workspace/EviMed/IMIS_data_jsons_cleaned/ObjectHallucinationBenchmark.json" \
  --val_omni_path="/workspace/EviMed/OmniMedVQA/QA_information/OmniMedVQA_1000.json" \
  --val_image_folder="/workspace/EviMed/OmniMedVQA/" \
  --image_folder='/workspace/EviMed/IMIS_data/' \
  --sub_image_folder="/workspace/EviMed/IMIS_data/" \
  --imis_sam_checkpoint="/workspace/EviMed/huggingface/IMISNet-B.pth" \
  --log_base_dir="/workspace/EviMed/Mobius_runs" \
  --vision_pretrained="/workspace/EviMed/huggingface/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=2 \
  --early_stop_vlm_step 70 \
  --early_stop_sam_step 100000 \
  --batch_size=8 \
  --workers=8 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --grad_accumulation_steps 5 \
  --imis_image_size 1024 \
  --out_dim 768 \
  --vlm_ce_loss_weight 1.0 \
  --sam_ce_loss_weight 0.1 \
  --vlm_mask_loss_weight 0.1 \
  --sam_mask_loss_weight 1.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_target_modules "gate_proj,up_proj,down_proj,q_proj,v_proj" \
  --sft_modules_vlm "lm_head,text_hidden_fcs" \
  --sft_modules_sam "mask_decoder,imis_to_vlm_proj_fcs" \
  --lr 0.00003 \
  --imis_lr 0.00005 \
  --warmup_ratio 0.03 \
  --save_steps 10 \
  --eval_steps 4 \
  2>&1 | tee -a $exp_dir/$time.log
