# vertex_translate.py
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# 환경 및 모델 초기화
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "env/gmail-361002-cbcf95afec4a.json"
vertexai.init(project="gmail-361002", location="us-central1")
# model = GenerativeModel(model_name='gemini-1.5-pro-002')
model = GenerativeModel(model_name='gemini-1.5-flash-002')

def translate_english_to_korean(english_sentence: str) -> str:
    """영어 문장을 한글로 번역한 결과만 반환하는 함수"""
    prompt = f"""다음 영어 문장을 한국어로 자연스럽게 번역해 주세요. 번역 결과만 출력하세요.

영어 문장: {english_sentence}

한글 번역:"""
    response = model.generate_content(prompt)
    print("번역 응답 : " , response.text.strip())
    return response.text.strip()
