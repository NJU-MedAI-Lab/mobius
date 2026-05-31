
#!/bin/bash
export PYTHONPATH=/workspace/EviMed/Mobius-code:$PYTHONPATH
# CUDA_VISIBLE_DEVICES="3"
gpu_list="2,3"
echo GPU_LIST: $gpu_list
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo CHUNKS: $CHUNKS

CKPT="mobius-stage3-loop-alllong-bs48-3e-5-5e-5-trainfcs-gradaccu5-vlmearlystop70-60"
NAME="mobius-stage3"

####################data config################
### For BTCV test
ROOT_PATH="/workspace/EviMed/IMIS_data_jsons_cleaned"

DATA_TYPE="ObjectHallucination_new_noregion_stage3"
SPLIT="ObjectHallucinationBenchmark"

### Control the answer type
answer_type='closed'
# answer_type='open'

PORT=64289
for IDX in $(seq 0 $((CHUNKS-1))); do
    PORT=$((PORT-1))
    deepspeed --include=localhost:${GPULIST[$IDX]} --master_port=$PORT model/eval/vqa_infer_benchmarks.py \
    --benchmark_name="hallucination" \
    --version="/workspace/EviMed/checkpoints/$CKPT" \
    --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
    --pretrain_mm_mlp_adapter="/workspace/EviMed/huggingface/llava-llama-2-7b-chat-clip336-pretrain-mobius/mm_projector.bin" \
    --answer_type=$answer_type \
    --image_folder='/workspace/EviMed/IMIS_data/' \
    --vision_pretrained="/workspace/EviMed/huggingface/sam-med2d_b.pth" \
    --imis_sam_checkpoint="/workspace/EviMed/huggingface/IMISNet-B.pth" \
    --val_data_path $ROOT_PATH/$SPLIT.json \
    --answers-file /workspace/EviMed/IMIS_tests/$NAME/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl \
    --workers 12 \
    --out_dim 768 \
    --max_sample_point 512 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &
done

wait

output_file=/workspace/EviMed/IMIS_tests/$NAME/$DATA_TYPE/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /workspace/EviMed/IMIS_tests/$NAME/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done