#!/bin/bash
NCCL_DEBUG=WARN
time=$(date +%Y-%m-%d-%H-%M-%S)
exp_name="mobius-stage1-Long+short+seg+other-VQA"
exp_dir="/workspace/EviMed/Mobius_runs/$exp_name"
mkdir -p "$exp_dir"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=65002 train_ds_mobius.py \
  --version="/workspace/EviMed/huggingface/llava-v1.5-7b/" \
  --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
  --pretrain_mm_mlp_adapter='/workspace/EviMed/huggingface/llava-llama-2-7b-chat-clip336-pretrain-mobius/mm_projector.bin' \
  --data_path="/workspace/EviMed/IMIS_data_jsons_cleaned/Stage1_Long+Short+Seg+OtherVQA-cleaned.json" \
  --val_data_path="/workspace/EviMed/IMIS_data_jsons_cleaned/Stage1_Long+Short+Seg+OtherVQA-cleaned.json" \
  --val_hallucination_path="/workspace/EviMed/IMIS_data_jsons_cleaned/ObjectHallucinationBenchmark.json" \
  --val_omni_path="/workspace/EviMed/OmniMedVQA/QA_information/OmniMedVQA_1000.json" \
  --val_image_folder="/workspace/EviMed/OmniMedVQA/" \
  --image_folder='/workspace/EviMed/IMIS_data/' \
  --sub_image_folder="/workspace/EviMed/IMIS_data/" \
  --imis_sam_checkpoint="/workspace/EviMed/huggingface/IMISNet-B.pth" \
  --log_base_dir="/data/Mobius_runs" \
  --exp_name=$exp_name \
  --epochs=4 \
  --batch_size=48 \
  --workers=16 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 2048 \
  --imis_image_size 1024 \
  --out_dim 768 \
  --grad_accumulation_steps 3 \
  --vlm_ce_loss_weight 1.0 \
  --lora_r 16 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --sft_modules_vlm "lm_head,embed_tokens,input_layernorm,post_attention_layernorm,mm_projector" \
  --lr 0.0005 \
  --save_steps 100 \
  --eval_steps 100 \
  --vlm_only \
  2>&1 | tee -a $exp_dir/$time.log