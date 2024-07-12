################################################################################
# Library Import
################################################################################

# LLM 관련
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer

# LLaVA 관련
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# 에러 관련
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 기타
import requests
from PIL import Image
from io import BytesIO

################################################################################
# 모델 업로드
################################################################################

model_path = "KBNIT/KoLLaVA-1.5v-kolon-v1.2" # Base model : "tabtoyou/KoLLaVA-v1.5-Synatra-7b", Fine-Tuned model : KBNIT/KoLLaVA-1.5v-kolon-v1.2

model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="balanced") # 모델 업로드
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, torch_dtype=torch.bfloat16) # 토크나이저 업로드

################################################################################
# Vision 설정
################################################################################

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor # 이미지 처리기 할당

################################################################################
# 추론 함수 설정
################################################################################

def caption_image(image_file, prompt):

    # 이미지 파일 불러오기
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # Torch 초기화 비활성화
    disable_torch_init()

    # 대화 템플릿 및 역할 설정
    conv_mode = "mistral"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # 이미지 전처리 및 텐서로 변환
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda() #half()
    image_tensor = image_tensor.type(torch.bfloat16)
  
    # 입력 생성 및 대화에 추가
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    # 토큰화 및 생성을 위한 입력 데이터 준비
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 생성 중단 기준 설정
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Torch 추론 모드에서 생성
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2, max_new_tokens=1024, 
                                     use_cache=True, top_k=50, top_p=0.80, min_length=10,  max_length=100, penalty_alpha=0.6, stopping_criteria=[stopping_criteria], no_repeat_ngram_size=5)

    # do_sample=True : 컨텍스트가 동일한 경우 다른 문장을 나오게 하기 위함.
    # min_length=10, max_length=50 : 생성된 문장이 너무 짧거나 길지 않게 하기 위함.
    # repetition_penalty=1.5, no_repeat_ngram_size=3 : 반복되는 토큰 제외
    # temperature=0.9 : 확률분포를 sharp하게 만들고 확률값이 높은 토큰이 더 잘 나오도록 설정.
    # top_k=50, top_p=0.92 : 탑k 샘플링, 탑p 샘플링을 동시에 적용해 확률값이 낮은 후보 단어는 배제하도록 설정
    # 기타 :  max_new_tokens=1024, stopping_criteria=[stopping_criteria]

    # 출력 디코딩 및 정리
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
  
    #conv.messages[-1][-1] = outputs
    #output = outputs.rsplit('</s>', 1)[0]
    print(outputs)
    return outputs

################################################################################
# 추론 
################################################################################
caption_image("./data/image_folder/train2014/COCO_train2014_000000000009.jpg", "이미지에서 어떤 종류의 음식을 볼 수 있나요?")


