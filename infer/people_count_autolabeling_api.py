import cv2
import requests
import base64
import numpy as np
import argparse
import os
from tqdm import tqdm

API_URL = "http://localhost:8000/predict_json" 

def process_video_ebc(video_path: str, time_interval: int):
    """
    비디오를 프레임 단위로 읽어 API로 추론하고, 결과를 영상에 지속적으로 그려넣습니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: 비디오 파일 '{video_path}'을(를) 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = f"{os.path.splitext(video_path)[0]}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"🚀 비디오 처리를 시작합니다: {video_path}")
    print(f"   - 총 프레임: {total_frames}, FPS: {fps:.2f}")
    print(f"   - {time_interval} 프레임마다 추론을 수행합니다.")
    print(f"   - 결과는 '{output_filename}' 파일에 저장됩니다.")
    
    # --- 변경점 1: 마지막 추론 결과를 저장할 변수 초기화 ---
    latest_text = None

    for frame_index in tqdm(range(total_frames), desc="처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        # time_interval 간격마다 API 추론 수행
        if frame_index % time_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_data = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "video_name": os.path.basename(video_path),
                "dense_dot": False,
                "frames": [{"frame_index": frame_index, "data": base64_data}]
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                if response.status_code == 200:
                    result_data = response.json()
                    score = result_data.get("results", [{}])[0].get("result", 0.0)
                    
                    # --- 변경점 2: 로컬 변수 text 대신 latest_text 변수를 업데이트 ---
                    latest_text = f"Result: {score:.4f}"
                    
                else:
                    print(f"\n⚠️ 프레임 {frame_index}: API 오류 (상태 코드: {response.status_code})")

            except requests.exceptions.RequestException as e:
                print(f"\n❌ 프레임 {frame_index}: API 연결 오류 - {e}")

        # --- 변경점 3: API 호출 여부와 상관없이, 매 프레임마다 저장된 텍스트를 그리기 ---
        # latest_text에 값이 있을 때만 (즉, 첫 추론 성공 후부터) 텍스트를 그립니다.
        if latest_text:
            position = (width - 300, 50) 
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0) # 녹색
            thickness = 2
            cv2.putText(frame, latest_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # 처리된 프레임을 새 비디오에 쓰기
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 처리 완료! 결과가 '{output_filename}'에 저장되었습니다.")

