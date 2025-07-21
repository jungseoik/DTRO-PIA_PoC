import cv2
import glob
import os
from datetime import timedelta
from tqdm import tqdm
import pandas as pd

from utils.api.vqa_api import internvl_vision_api_response_vqa

__all__ = [
    "process_video",
    "batch_process_videos",
]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def frame_to_timecode(frame_idx: int, fps: float) -> str:
    """Convert frame index → HH:MM:SS.ff time‑code string."""
    seconds = frame_idx / fps
    return str(timedelta(seconds=seconds)).split(".")[0] + f".{int((seconds % 1)*100):02d}"


def wrap_text_cv(text: str, font_face, font_scale: float, thickness: int, max_width: int):
    """Simple word‑wrap for OpenCV text rendering."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        (tw, _), _ = cv2.getTextSize(test, font_face, font_scale, thickness)
        if tw <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# -----------------------------------------------------------------------------
# Core single‑video processing
# -----------------------------------------------------------------------------

def process_video(
    video_path: str,
    time_interval: int,
    question: str,
    output_root_dir: str,
):
    """Sample *every* `time_interval` frames without skipping duplicates.

    Results (sampled frame → image, subtitle‑burned video, CSV) are saved in
    `<output_root_dir>/<video_name>/`.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir   = os.path.join(output_root_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 결과 영상 라이터
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    vid_out  = cv2.VideoWriter(
        os.path.join(out_dir, f"{base_name}_subtitled.mp4"),
        fourcc, fps, (width, height)
    )

    # 글꼴 & 렌더 설정
    font_face         = cv2.FONT_HERSHEY_SIMPLEX
    cat_font_scale    = 0.9
    desc_font_scale   = 1.0
    font_thick        = 2
    line_spacing_px   = 10
    alpha_bg          = 0.6
    max_desc_width_px = int(width * 0.8)

    # 상태 저장
    latest_category    = None
    latest_description = None
    csv_records        = []

    for idx in tqdm(range(n_frames), desc=base_name):
        success, frame = cap.read()
        if not success:
            break

        # ───────────────────────────────────────────────────────── frame sampling
        if idx % time_interval == 0:
            category, description = internvl_vision_api_response_vqa(frame, question)

            # 안전장치 – 둘 다 None 일 경우 대비
            category     = category or "unknown"
            description  = description or ""

            latest_category    = category
            latest_description = description

            time_code = frame_to_timecode(idx, fps)
            alarm     = 0 if category.lower() == "normal" else 1

            # CSV 로그 및 샘플 이미지 저장 (항상 저장!)
            csv_records.append({
                "frame": idx,
                "time":  time_code,
                "description": description,
                "category": category,
                "alarm": alarm,
            })

            img_name = f"{base_name}_{idx}_{category}.jpg"
            cv2.imwrite(os.path.join(out_dir, img_name), frame)

        # ───────────────────────────────────────────────────────── subtitle overlay
        if latest_category is not None:
            # 1) Category 상단 박스
            cat_text = latest_category.upper()
            (tw, th), _ = cv2.getTextSize(cat_text, font_face, cat_font_scale, font_thick)
            x1, y1 = (width - tw) // 2 - 10, 20
            x2, y2 = x1 + tw + 20, y1 + th + 20
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha_bg, frame, 1 - alpha_bg, 0, frame)

            color = (255, 255, 255) if latest_category.lower() == "normal" else (0, 0, 255)
            txt_x, txt_y = (width - tw) // 2, y1 + th + 5 - 10
            cv2.putText(frame, cat_text, (txt_x, txt_y), font_face, cat_font_scale, color, font_thick, cv2.LINE_AA)

            # 2) Description 하단 박스
            desc_lines = wrap_text_cv(latest_description, font_face, desc_font_scale, font_thick, max_desc_width_px)
            if desc_lines:
                sizes = [cv2.getTextSize(line, font_face, desc_font_scale, font_thick)[0] for line in desc_lines]
                bw    = max(w for w, _ in sizes)
                bh    = sum(h for _, h in sizes) + line_spacing_px * (len(desc_lines) - 1)
                x1, y1 = (width - bw) // 2 - 10, height - bh - 30
                x2, y2 = x1 + bw + 20, y1 + bh + 20

                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, alpha_bg, frame, 1 - alpha_bg, 0, frame)

                y_text = y1 + 10 + sizes[0][1]
                for (w, h), line in zip(sizes, desc_lines):
                    x_text = (width - w) // 2
                    cv2.putText(frame, line, (x_text, y_text), font_face, desc_font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
                    y_text += h + line_spacing_px

        vid_out.write(frame)

    # --------------------------------------------------------------------- wrap‑up
    cap.release()
    vid_out.release()

    # CSV 저장
    pd.DataFrame(csv_records).to_csv(
        os.path.join(out_dir, f"{base_name}_vqa_alarm_descrition_result.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"✅ Finished: {video_path} → {out_dir}")

# -----------------------------------------------------------------------------
# Batch helper
# -----------------------------------------------------------------------------

def batch_process_videos(
    input_path: str,
    time_interval: int,
    question: str,
    output_root_dir: str,
    exts: tuple[str, ...] = ("*.mp4", "*.avi", "*.mov", "*.mkv"),
):
    """Process single file or recursively all videos inside a folder."""
    if os.path.isfile(input_path):
        process_video(input_path, time_interval, question, output_root_dir)
        return

    # 폴더 처리
    videos: list[str] = []
    for pattern in exts:
        videos.extend(glob.glob(os.path.join(input_path, pattern)))
    videos.sort()

    if not videos:
        print("⚠️  No video files found.")
        return

    for vid in videos:
        try:
            process_video(vid, time_interval, question, output_root_dir)
        except Exception as e:
            print(f"❌ {os.path.basename(vid)}: {e}")

# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="Video VQA sampler (single‑output folder)")
    pa.add_argument("input", help="Video file or folder")
    pa.add_argument("-o", "--output", required=True, help="Output root directory")
    pa.add_argument("-q", "--question", default="Describe the scene", help="Question passed to VQA model")
    pa.add_argument("-t", "--interval", type=int, default=30, help="Sampling interval in frames")
    args = pa.parse_args()

    batch_process_videos(args.input, args.interval, args.question, args.output)
