# Möbius: Bridging Mutual Supervision and Compensation between LVLMs and SAMs for Robust Medical Vision-Language Systems

## 🔧 Dependencies and Installation
- Python == 3.10.18
- [PyTorch == 2.1.2+cu118](https://pytorch.org/)
- [transformers >= 4.31.0](https://huggingface.co/docs/transformers)
- [deepspeed == 0.31.1](https://deepspeed.readthedocs.io/en/latest/)
- [xformers == 0.0.23.post1+cu118](https://github.com/facebookresearch/xformers.git)

### Installation
1. Clone repo

```bash
git clone https://github.com/NJU-MedAI-Lab/Mobius.git
cd Mobius
```
2. Install dependent packages (use conda)

```bash
conda create --name mobius python==3.10.18 -y
conda activate mobius
pip install -r requirements.txt
```

## 🗂️ Datasets

Our datasets are build based on [IMed-361M](https://huggingface.co/datasets/General-Medical-AI/IMed-361M) dataset.

- You can download the MedSegVQA dataset from [Google Drive](https://drive.google.com/drive/folders/1iDKKebav4selHiOl0y2aaietQh6murR3?usp=sharing) and download the images from [Huggingface](https://huggingface.co/datasets/General-Medical-AI/IMed-361M).

- You can download the MedHOPE benchmark and images from [Google Drive](https://drive.google.com/drive/folders/1g9FW7AbZJynWRAIoWjqjpAFE57DYbz5o?usp=drive_link).

## ⚙️ Train

### Stage 1

1. We perfrom the pre-training stage 1 to get the projector checkpoints. Please obtain the llava_med_alignment_500k dataset according to [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), and then follow the usage tutorial of [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA) to pretrain.

2. After pre-training, we use our MedSegVQA dataset to fully fine-tune LLaVA-v1.5.
```bash
bash scripts/train_stage1.sh
```

### Stage 2

```bash
bash scripts/train_stage2.sh
```

### Stage 3

1. We pre-trained the Text2Image projector firstly using a subset of MedSegVQA-TS.
```bash
bash scripts/train_stage3_proj.sh
```

2. Then train our Mobius framework.
```bash
bash scripts/train_stage3.sh
```

## 🏰 Model Zoo

We provide Mobius checkpoints on [Huggingface](https://huggingface.co/henry984/Mobius)

## ⚡️ Test

### Open-set VQA
```bash
TRANSFORMERS_OFFLINE=1 deepspeed --include=localhost:0,1 --master_port=64995 model/eval/vqa_infer.py \
    --version="/path/to/the/mobius_checkpoints" \
    --vision_tower='/path/to/the/clip-vit-large-patch14-336' \
    --pretrain_mm_mlp_adapter='/path/to/the/pretrained-Image2Text-adapter' \
    --answer_type='open' \
    --val_data_path='/path/to/the/openVQA_json_file' \
    --image_folder='/path/to/the/IMed-361M' \
    --vision_pretrained='/path/to/the/sam-med2d_b.pth' \
    --imis_sam_checkpoint='/path/to/the/IMISNet-B.pth' \
    --eval_vqa \
    # --vis_mask \
```

### Text-prompted Segmentation
Infer to generate the prediction jsonl file
```bash
bash model/eval/infer_parallel_seg.sh
```

Calculate the metrics
```py
python model/eval/cal_metric_seg.py
```

### Visual Grounding
Infer to generate the prediction jsonl file
```bash
bash model/eval/infer_parallel_grounding.sh
```

Calculate the metrics
```py
python model/eval/cal_metric_openvqa_from_grounding.py
```

### Region-aware VQA
Infer to generate the prediction jsonl file
```bash
bash model/eval/infer_parallel_openvqa.sh
```

Calculate the metrics
```py
python model/eval/cal_metric_openvqa.py
```

### MedHOPE Benchmark
Infer to generate the prediction jsonl file
```bash
bash model/eval/infer_parallel_hallucination.sh
```

Calculate the metrics
```py
python model/eval/cal_metric_objecthallucination.py
```

### OmniMedVQA Benchmark
Infer to generate the prediction jsonl file
```bash
bash model/eval/infer_parallel_omniMedVQA.sh
```

Calculate the metrics
```py
python model/eval/cal_metric_omniMedVQA.py
```