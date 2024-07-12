# KoLLaVA
Vision LLM인 KoLLaVA에 대한 학습, 추론 관련 코드입니다.

## 1. Fine-tuning 
학습은 LLaVA 1.5 모델을 한국어 버전으로 학습한 tabtoyou/KoLLaVA-v1.5-Synatra-7b를 베이스 모델로 학습했습니다. \
학습에는 LoRA 방법론이 사용되어 학습 후 반드시 Adapter를 수행하여 베이스 모델과 merge 해야 합니다. \
학습은 KoLLaVA Path에서 다음의 코드로 수행했습니다. 학습 파라미터는 LLM 파라미터와 유사하니 확인 후 수정해주세요. 

deepspeed ./llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed "./scripts/zero3.json" \
    --model_name_or_path "tabtoyou/KoLLaVA-v1.5-Synatra-7b" \
    --version mistral \
    --data_path "./data/ko_coco2014_train.json" \
    --image_folder "./data/image_folder/train2014" \
    --vision_tower "openai/clip-vit-large-patch14-336" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "./checkpoints/KoLLaVA-1.5v-lora-kolon-v1.2" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard 

## 2. LoRA Merge
Fine-tuning은 LoRA 기법으로 이루어지며, 생성된 LoRA 파일과 베이스 모델 간 Merge가 필요합니다.
- [[Adapter]KoLLaVA.ipynb](https://github.com/Kolon-Benit/KoLLaVA/blob/main/%5BAdapter%5DKoLLaVA.ipynb)
- [[Adapter]KoLLaVA.py](https://github.com/Kolon-Benit/KoLLaVA/blob/main/%5BAdapter%5DKoLLaVA.py)

## 3. Inference
허깅페이스에 업로드 되거나, 학습한 모델을 추론하는 코드입니다.
- [[Inference]KoLLaVA.ipynb](https://github.com/Kolon-Benit/KoLLaVA/blob/main/%5BInference%5DKoLLaVA.ipynb)
- [[Inference]KoLLaVA.py](https://github.com/Kolon-Benit/KoLLaVA/blob/main/%5BInference%5DKoLLaVA.py)
  
## 출처 
[KoLLaVA](https://github.com/tabtoyou/KoLLaVA.git) 를 기반으로 수정한 코드입니다.

## 주의사항
해당 모델을 활용할 시 출처 모델의 라이센스를 확인해주세요. 
