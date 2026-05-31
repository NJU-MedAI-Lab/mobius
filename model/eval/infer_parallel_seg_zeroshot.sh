
#!/bin/bash
export PYTHONPATH=/workspace/EviMed/Mobius-code:$PYTHONPATH
# CUDA_VISIBLE_DEVICES="3"
gpu_list="2"
echo GPU_LIST: $gpu_list
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo CHUNKS: $CHUNKS

CKPT="mobius-stage3-loop-alllong-bs48-3e-5-5e-5-trainfcs-gradaccu5-vlmearlystop70-60"

####################data config################
### For BTCV test
ROOT_PATH="/workspace/EviMed/SA-Med-2D-20M/"

DATA_TYPE="ISLES2022"
SPLIT="dataset_test"

### Control the answer type
# answer_type='closed'
answer_type='open'

PORT=64944
for IDX in $(seq 0 $((CHUNKS-1))); do
    PORT=$((PORT-1))
    deepspeed --include=localhost:${GPULIST[$IDX]} --master_port=$PORT model/eval/vqa_infer.py \
    --version="/workspace/EviMed/checkpoints/$CKPT" \
    --pretrain_sam="/workspace/EviMed/checkpoints/$CKPT" \
    --vision_tower='/workspace/EviMed/huggingface/clip-vit-large-patch14-336' \
    --answer_type=$answer_type \
    --image_folder='/workspace/EviMed/SA-Med-2D-20M/' \
    --vision_pretrained="/workspace/EviMed/huggingface/sam-med2d_b.pth" \
    --val_data_path $ROOT_PATH/$DATA_TYPE/$SPLIT.json \
    --vis_save_path /workspace/EviMed/Mobius_tests/$CKPT/$DATA_TYPE/vis-imgs/ \
    --answers-file /workspace/EviMed/Mobius_tests/$CKPT/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl \
    --workers 12 \
    --eval_seg \
    --vis_mask \
    --out_dim 768 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &
done

wait

output_file=/workspace/EviMed/Mobius_tests/$CKPT/$DATA_TYPE/$SPLIT.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /workspace/EviMed/Mobius_tests/$CKPT/$DATA_TYPE/$SPLIT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

