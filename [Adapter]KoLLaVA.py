################################################################################
# Library Import
################################################################################

# LLM 관련
from transformers import AutoTokenizer, AutoModelForCausalLM

# 허깅페이스 관련
from huggingface_hub import login
from huggingface_hub import create_repo

# LLaVA 관련
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# 기타
import argparse

################################################################################
# 모델 업로드
################################################################################

model_name = "KoLLaVA-1.5v-lora-kolon-v1.2" # Fine-tuning 된 모델의 Adapter
model_base = "tabtoyou/KoLLaVA-v1.5-Synatra-7b" # Base 모델

model_path = f"./checkpoints/{model_name}" # Adapther 경로
save_model_path = f"./checkpoints/merged/{model_name}" # Adapter가 연결된 최종 LLM 모델의 저장 경로
save_model_path_hf = f"KBNIT/{model_name}" # 최종 LLM 모델을 저장할 허깅페이스 경로

################################################################################
# Adapter 연결 함수 설정
################################################################################

def merge_lora(model_path, model_base, save_model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, device_map='cpu')

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
  
    login(token = "토큰을 입력하세요.") # 허깅페이스 토큰 입력
    create_repo(save_model_path_hf) # 허깅페이스 Repository 생성

    model.save_pretrained(save_model_path_hf, push_to_hub=True)
    tokenizer.save_pretrained(save_model_path_hf, push_to_hub=True)

################################################################################
# Adapter 연결 
################################################################################

merge_lora(model_path, model_base, save_model_path)