import cv2
import base64
import requests
import json
import os
from PIL import Image
import io
import numpy as np

def frame_to_b64(frame):
    success, buf = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Encode failed")
    return base64.b64encode(buf.tobytes()).decode()

def save_b64_png(data_uri: str, out_path: str):
    """
    data_uri: "data:image/png;base64,XXXX..."
    out_path: 저장할 파일 경로 (예: "dense_0.png")
    """
    header, b64 = data_uri.split(",", 1)
    img_bytes = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(img_bytes)

if __name__ == "__main__":
    # --- 1) 프레임 준비 & 요청 ---
    image_path = "assets/289.jpg"
    image_path = "/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/batch_test_sample.png"
    frame = np.array(Image.open(image_path).convert("RGB"))

    payload_frames = [{
        "frame_index": 0,
        "data": frame_to_b64(frame)
    }]

    payload = {
        "video_name": "my_video.mp4",
        # dense_dot 옵션을 켜려면 아래 주석 해제
        "dense_dot": True,
        "frames": payload_frames
    }

    resp = requests.post(
        "http://localhost:8000/predict_json",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    resp.raise_for_status()
    result = resp.json()

    # --- 2) 결과 저장 디렉토리 생성 ---
    out_dir = "predict_output"
    os.makedirs(out_dir, exist_ok=True)

    # --- 3) 응답 처리 & 이미지 저장 ---
    for item in result["results"]:
        idx = item["frame_index"]
        count = item["result"]
        print(f"Frame {idx}: count = {count}")

        # dense_map 이 있을 때
        if item.get("dense_map"):
            out_path = os.path.join(out_dir, f"dense_{idx}.png")
            save_b64_png(item["dense_map"], out_path)
            print(f"  → Dense map saved: {out_path}")

        # dot_map 이 있을 때
        if item.get("dot_map"):
            out_path = os.path.join(out_dir, f"dot_{idx}.png")
            save_b64_png(item["dot_map"], out_path)
            print(f"  → Dot map   saved: {out_path}")

    print("모든 이미지가 저장되었습니다.")
