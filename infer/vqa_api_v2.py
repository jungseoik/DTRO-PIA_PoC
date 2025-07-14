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

def process_video(video_path: str, time_interval: int, question: str, lang: str = "en"):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = f"{os.path.splitext(video_path)[0]}_subtitled.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    latest_sub = None
    latest_sub_ko = None
    # 공통
    max_w    = int(width * 0.8)
    alpha_bg = 0.6

    # 영어(OpenCV) 설정
    font_cv    = cv2.FONT_HERSHEY_SIMPLEX
    scale_cv   = 1.0
    thick_cv   = 2
    line_sp_cv = 10

    # 한글(PIL) 설정
    # NanumGothic.ttf 같은 한글 폰트 파일을 프로젝트 경로에 두세요
    pil_font = ImageFont.truetype("NanumGothic.ttf", 32)
    line_sp_pil = 4

    for i in tqdm(range(frames), desc="처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 일정 간격마다 API 호출
        if i % time_interval == 0:
            success, buf = cv2.imencode('.jpg', frame)
            if success:
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
                try:
                    r = requests.post(API_URL, json=payload, timeout=30)
                    if r.status_code == 200:
                        sub = parse_response(r.json())
                        if sub and sub != latest_sub:
                            latest_sub = sub
                            if lang == "kor":
                                latest_sub_ko = translate_english_to_korean(latest_sub)
                except:
                    pass

        # 2) 캡션 그리기
        if latest_sub:
            if lang == 'en':
                # --- 영어: OpenCV 로 그리기 ---
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

            elif lang == "kor":
                text = latest_sub_ko
                # 1) Frame → PIL
                img_pil = Image.fromarray(frame).convert("RGBA")
                draw    = ImageDraw.Draw(img_pil)

                # 2) 번역 (필요 시) + 줄바꿈
                lines  = wrap_text_pil(text, pil_font, max_w, line_sp_pil)
                
                # 3) 텍스트 블록 크기 계산
                # multiline_textbbox 로 전체 박스 측정
                text_block = "\n".join(lines)
                bbox = draw.multiline_textbbox((0,0), text_block, font=pil_font, spacing=line_sp_pil)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                # 4) 위치 계산 (하단 중앙)
                x = (width - tw)//2
                y = height - th - 30

                # 5) 반투명 배경
                overlay = Image.new("RGBA", img_pil.size, (0,0,0,0))
                bg = Image.new("RGBA", (tw+20, th+20), (0,0,0, int(255*alpha_bg)))
                overlay.paste(bg, (x-10, y-10))
                img_pil = Image.alpha_composite(img_pil, overlay)

                # 6) 텍스트 그리기
                draw = ImageDraw.Draw(img_pil)
                draw.multiline_text((x, y), text_block,
                                    font=pil_font,
                                    fill=(255,255,255,255),
                                    spacing=line_sp_pil,
                                    align="center")

                # 7) PIL → numpy
                frame = np.array(img_pil.convert("RGB"))

        out.write(frame)

    cap.release()
    out.release()
    print(f"\n✅ 저장 완료: {out_path}")

