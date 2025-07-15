import cv2
import requests
import base64
import argparse
import os
from tqdm import tqdm
import numpy as np

# PIL은 한글 자막에만 사용
from PIL import Image, ImageDraw, ImageFont
import textwrap
from translator.vertex_translate import translate_english_to_korean
from utils.api.vqa_api import internvl_vision_api_response_vqa
# --- 설정 ---
API_URL    = "http://127.0.0.1:9997/v1/chat/completions"
MODEL_NAME = "InternVL3"
MAX_TOKENS = 256

def parse_response(response_data: dict) -> str | None:
    try:
        return response_data["choices"][0]["message"]["content"].strip()
    except:
        return None

def wrap_text_cv(text, font, scale, thickness, max_width):
    # OpenCV 전용 줄바꿈
    words, lines, cur = text.split(' '), [], ''
    for w in words:
        test = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

def wrap_text_pil(text, font, max_width, spacing):
    """
    PIL 이미지 드로어(draw)를 쓰지 않고도
    글자 단위로 최대 픽셀 너비를 넘지 않게 줄바꿈하는 함수.
    """
    words = text.split(' ')
    lines = []
    current = ''
    for w in words:
        test = f"{current} {w}".strip()
        # 임시 이미지로 크기 재기
        tmp_img = Image.new("RGB", (1,1))
        tmp_draw = ImageDraw.Draw(tmp_img)
        bbox = tmp_draw.textbbox((0,0), test, font=font, spacing=spacing)
        width_px = bbox[2] - bbox[0]
        if width_px <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

def process_video_v3(video_path: str, time_interval: int, question: str):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{os.path.splitext(video_path)[0]}_subtitled_v3.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    latest_category = None
    latest_sub = None
    # 공통
    # 공통
    max_w    = int(width * 0.8)
    alpha_bg = 0.6

    # 영어(OpenCV) 설정
    font_cv    = cv2.FONT_HERSHEY_SIMPLEX
    scale_cv   = 1.0
    thick_cv   = 2
    line_sp_cv = 10

    # 영어(OpenCV) 설정
    font_cv    = cv2.FONT_HERSHEY_SIMPLEX

    for i in tqdm(range(frames), desc="처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 일정 간격마다 API 호출
        if i % time_interval == 0:
            category, description = internvl_vision_api_response_vqa(frame, question)
            if description and description != latest_sub:
                latest_sub = description
                latest_category = category

        if latest_category:
            cat_text = latest_category.upper()
            font_cat_scale = 0.9
            font_cat_thick = 2

            (tw, th), _ = cv2.getTextSize(cat_text, font_cv, font_cat_scale, font_cat_thick)

            # 상단 중앙 좌표
            x1 = (width - tw)//2 - 10
            y1 = 20
            x2 = x1 + tw + 20
            y2 = y1 + th + 20

            # 배경
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,0,0), -1)
            cv2.addWeighted(overlay, alpha_bg, frame, 1-alpha_bg, 0, frame)

            # 글자 색상 결정
            if latest_category.lower() == "normal":
                color = (255, 255, 255)  # 하얀 글자
            else:
                color = (0, 0, 255)      # 빨간 글자

            # 텍스트
            text_x = (width - tw) // 2
            text_y = y1 + th + 5 - 10  # 살짝 아래로 보정
            cv2.putText(frame, cat_text, (text_x, text_y),
                        font_cv, font_cat_scale, color, font_cat_thick, cv2.LINE_AA)
            
            lines = wrap_text_cv(latest_sub, font_cv, scale_cv, thick_cv, max_w)
            sizes = [cv2.getTextSize(line, font_cv, scale_cv, thick_cv)[0] for line in lines]
            bw = max(w for w, h in sizes)
            bh = sum(h for w, h in sizes) + line_sp_cv*(len(lines)-1)

            x1 = (width - bw)//2 - 10
            y1 = height - bh - 30
            x2, y2 = x1 + bw + 20, y1 + bh + 20

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,0), -1)
            cv2.addWeighted(overlay, alpha_bg, frame, 1-alpha_bg, 0, frame)

            y = y1 + 10 + sizes[0][1]
            for (w, h), line in zip(sizes, lines):
                x = (width - w)//2
                cv2.putText(frame, line, (x, y), font_cv, scale_cv, (255,255,255), thick_cv, cv2.LINE_AA)
                y += h + line_sp_cv


        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ 저장 완료: {out_path}")

