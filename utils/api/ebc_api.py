import streamlit as st
import cv2
import os
import cv2
import time
import base64
import requests
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import LetterBox
import io
from PIL import Image
import numpy as np

MAX_WIDTH = 854
MAX_HEIGHT = 480
API_URL = "http://localhost:8000/predict_json" 


def b64_to_rgb_np(data_uri: str) -> np.ndarray:
    """
    data_uri: "data:image/png;base64,XXXX..."
    return: (H,W,3) RGB uint8 numpy array
    """
    header, b64 = data_uri.split(",", 1)
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)


def call_inference_api(video_name: str, frame_index: int, frame_bgr: np.ndarray, dense_dot=False, timeout=5):
    """
    단일 프레임에 대해 API 요청 및 응답 결과 반환
    """
    _, buf = cv2.imencode('.jpg', frame_bgr)
    b64 = base64.b64encode(buf).decode('utf-8')
    payload = {
        "video_name": video_name,
        "dense_dot": dense_dot,
        "frames": [{"frame_index": frame_index, "data": b64}]
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()["results"][0]
            return {
                "count": data["result"],
                "dense_map": b64_to_rgb_np(data["dense_map"]) if dense_dot else None,
                "dot_map": b64_to_rgb_np(data["dot_map"]) if dense_dot else None,
            }
        else:
            st.warning(f"API 오류 @frame {frame_index}: {resp.status_code}")
    except Exception as e:
        st.warning(f"API 연결 실패 @frame {frame_index}: {e}")
    
    return None
