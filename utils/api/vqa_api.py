import base64
import requests
import cv2
import json
import re
from typing import Tuple, Optional
# --- 설정 ---
API_URL = "http://127.0.0.1:9997/v1/chat/completions"
MODEL_NAME = "InternVL3"
MAX_TOKENS = 256

def parse_response(response_data: dict) -> str | None:
    try:
        return response_data["choices"][0]["message"]["content"].strip()
    except:
        return None

# 또는 OpenCV로 이미지를 읽어서 처리하고 싶다면:
def internvl_vision_api_response(image_path: str, question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    이미지 경로를 OpenCV로 읽어서 Vision API 응답을 반환하는 함수
    
    Args:
        image_path: 이미지 파일 경로
        question: 질문 텍스트
    
    Returns:
        Tuple[str|None, str|None]: (category, description) 또는 (None, None)
    """
    try:
        # OpenCV로 이미지 읽기
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return None, None
        
        # 이미지를 JPEG로 인코딩 후 base64 변환
        success, buf = cv2.imencode('.jpg', frame)
        if not success:
            return None, None
            
        b64 = base64.b64encode(buf).decode()
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<image>\n{question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            "max_tokens": MAX_TOKENS
        }
        
        # API 요청
        r = requests.post(API_URL, json=payload, timeout=30)
        if r.status_code == 200:
            api_response = parse_response(r.json())
            if api_response:
                return parse_vision_response(api_response)
            else:
                return None, None
        else:
            return None, None
            
    except Exception as e:
        print(f"API 요청 중 오류: {e}")
        return None, None
    

def parse_vision_response(result_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Vision API 응답을 파싱하여 category와 description을 추출하는 함수
    
    Args:
        result_text: API 응답 텍스트
    
    Returns:
        Tuple[str|None, str|None]: (category, description) 
        파싱 실패시 (None, None) 반환
    """
    if not result_text:
        return None, None

    cleaned = result_text.strip()

    # ```json 또는 ``` 제거
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```json|^```|```$", "", cleaned.strip(), flags=re.IGNORECASE).strip()

    # 불필요한 따옴표 정리
    cleaned = cleaned.replace('""', '"')

    category = None
    description = None

    try:
        # JSON 파싱 시도
        parsed = json.loads(cleaned)
        category = parsed.get("category")
        description = parsed.get("description")
        
    except json.JSONDecodeError:
        # JSON 파싱 실패시 정규식으로 추출
        
        # category 추출
        cat_match = re.search(r'"?category"?\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)
        if cat_match:
            category = cat_match.group(1)
        
        # description 추출
        desc_match = re.search(r'"?description"?\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1)

    return category, description

import numpy as np

def internvl_vision_api_response_vqa(image_input, question: str) -> Tuple[Optional[str], Optional[str]]:
    """
    이미지 경로 또는 numpy 배열을 받아서 Vision API 응답을 반환하는 함수
    
    Args:
        image_input: 이미지 파일 경로(str) 또는 numpy 배열(np.ndarray)
        question: 질문 텍스트
    
    Returns:
        Tuple[str|None, str|None]: (category, description) 또는 (None, None)
    """
    try:
        # 입력 타입에 따라 처리
        if isinstance(image_input, str):
            # 문자열인 경우 파일 경로로 처리
            frame = cv2.imread(image_input)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_input}")
                return None, None
        elif isinstance(image_input, np.ndarray):
            # numpy 배열인 경우 그대로 사용
            frame = image_input
        else:
            print(f"지원하지 않는 입력 타입: {type(image_input)}")
            return None, None
        
        # 이미지를 JPEG로 인코딩 후 base64 변환
        success, buf = cv2.imencode('.jpg', frame)
        if not success:
            print("이미지 인코딩 실패")
            return None, None
            
        b64 = base64.b64encode(buf).decode()
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<image>\n{question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }],
            "max_tokens": MAX_TOKENS
        }
        
        # API 요청
        r = requests.post(API_URL, json=payload, timeout=30)
        if r.status_code == 200:
            api_response = parse_response(r.json())
            if api_response:
                return parse_vision_response(api_response)
            else:
                return None, None
        else:
            print(f"API 요청 실패: {r.status_code}")
            return None, None
            
    except Exception as e:
        print(f"API 요청 중 오류: {e}")
        return None, None