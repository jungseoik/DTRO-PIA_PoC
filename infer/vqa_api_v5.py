import cv2
import requests
import base64
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
# PIL은 한글 자막에만 사용
from PIL import Image, ImageDraw, ImageFont
import textwrap
from translator.vertex_translate import translate_english_to_korean
from datetime import timedelta
import shutil
import csv

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


def process_video2(
    video_path: str,
    time_interval: int,
    question: str,
    lang: str = "en",
    result_root_dir: str = "/tmp/vqa_video_result"
):
    """
    한 개 비디오를 처리해
      <result_root_dir>/<비디오이름>/ 에
        · 자막 삽입 비디오
        · CSV
        · 추론 시점 프레임 이미지
    를 저장한다.
    """
    # ── 1. 비디오 메타 정보 ───────────────────────────────────────
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name  = os.path.splitext(os.path.basename(video_path))[0]
    save_dir   = os.path.join(result_root_dir, base_name)   # ★ 비디오마다 전용 폴더
    os.makedirs(save_dir, exist_ok=True)

    # ── 2. 출력 비디오 설정 ───────────────────────────────────────
    out_path = os.path.join(save_dir, f"{base_name}_subtitled_only_descrition.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # ── 3. 기타 준비 ──────────────────────────────────────────────
    records, latest_sub, latest_sub_ko = [], None, None
    max_w, alpha_bg = int(width*0.8), 0.6
    font_cv, scale_cv, thick_cv, line_sp_cv = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2, 10
    pil_font   = ImageFont.truetype("NanumGothic.ttf", 32)
    line_sp_pil = 4

    for i in tqdm(range(frames), desc=f"[{base_name}] 처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        # ───── 추론 타이밍 ─────
        if i % time_interval == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                b64 = base64.b64encode(buf).decode()
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{
                        "role": "user",
                        "content":[
                            {"type":"text","text":f"<image>\n{question}"},
                            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
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
                                latest_sub_ko = translate_english_to_korean(sub)

                            time_str = str(timedelta(seconds=i/fps))[:10]
                            records.append({"frame": i, "time": time_str, "description": sub})

                            # 프레임 이미지 저장
                            cv2.imwrite(os.path.join(save_dir, f"{base_name}_{i}.jpg"), frame)
                except Exception as e:
                    print(f"[ERROR] {base_name} frame {i}: {e}")

        # ───── 자막 렌더링 ─────
        if latest_sub:
            if lang == "en":
                lines  = wrap_text_cv(latest_sub, font_cv, scale_cv, thick_cv, max_w)
                sizes  = [cv2.getTextSize(l, font_cv, scale_cv, thick_cv)[0] for l in lines]
                bw, bh = max(w for w,_ in sizes), sum(h for _,h in sizes)+line_sp_cv*(len(lines)-1)

                x1, y1 = (width-bw)//2-10, height-bh-30
                x2, y2 = x1+bw+20, y1+bh+20
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,0,0), -1)
                cv2.addWeighted(overlay, alpha_bg, frame, 1-alpha_bg, 0, frame)
                y = y1 + 10 + sizes[0][1]
                for (w,h), line in zip(sizes, lines):
                    cv2.putText(frame, line, ((width-w)//2, y), font_cv, scale_cv, (255,255,255), thick_cv, cv2.LINE_AA)
                    y += h + line_sp_cv

            elif lang == "kor":
                text     = latest_sub_ko
                img_pil  = Image.fromarray(frame).convert("RGBA")
                draw     = ImageDraw.Draw(img_pil)
                lines    = wrap_text_pil(text, pil_font, max_w, line_sp_pil)
                block    = "\n".join(lines)
                bbox     = draw.multiline_textbbox((0,0), block, font=pil_font, spacing=line_sp_pil)
                tw, th   = bbox[2]-bbox[0], bbox[3]-bbox[1]
                x, y     = (width-tw)//2, height-th-30

                overlay  = Image.new("RGBA", img_pil.size, (0,0,0,0))
                bg       = Image.new("RGBA", (tw+20, th+20), (0,0,0,int(255*alpha_bg)))
                overlay.paste(bg, (x-10, y-10))
                img_pil  = Image.alpha_composite(img_pil, overlay)
                draw     = ImageDraw.Draw(img_pil)
                draw.multiline_text((x,y), block, font=pil_font, fill=(255,255,255,255),
                                    spacing=line_sp_pil, align="center")
                frame = np.array(img_pil.convert("RGB"))

        out.write(frame)

    cap.release()
    out.release()

    # ── 4. CSV 저장 ───────────────────────────────────────────────
    pd.DataFrame(records) \
      .to_csv(os.path.join(save_dir, f"{base_name}_subtitled_only_descrition.csv"),
              index=False, encoding="utf-8-sig")

    print(f"✅ {base_name} 완료  →  {save_dir}")

# ────────────────────────────────────────────────
# 폴더-단위 일괄 처리 래퍼
# ────────────────────────────────────────────────
import glob
def batch_process_videos2(
    folder_path: str,
    time_interval: int,
    question: str,
    lang: str = "en",
    result_root_dir: str = "/tmp/vqa_video_result",
    exts: tuple[str,...] = ("*.mp4","*.avi","*.mov","*.mkv")
):
    videos=[]
    for p in exts:
        videos.extend(glob.glob(os.path.join(folder_path, p)))
    videos.sort()
    if not videos:
        print("⚠️  처리할 비디오가 없습니다.")
        return

    print(f"🔍 {len(videos)}개 비디오 발견 — 순차 처리 시작")
    for vid in videos:
        try:
            process_video2(
                video_path      = vid,
                time_interval   = time_interval,
                question        = question,
                lang            = lang,
                result_root_dir = result_root_dir
            )
        except Exception as e:
            print(f"❌ '{os.path.basename(vid)}' 오류: {e}")
