import cv2
import requests
import base64
import argparse
import os
from tqdm import tqdm

# --- 설정 ---
API_URL = "http://127.0.0.1:9997/v1/chat/completions"
MODEL_NAME = "InternVL3"
MAX_TOKENS = 256

def parse_response(response_data: dict) -> str | None:
    """
    API 응답(JSON)을 파싱하여 모델의 답변 텍스트를 추출합니다.
    """
    try:
        content = response_data["choices"][0]["message"]["content"]
        return content.strip()
    except (KeyError, IndexError, TypeError) as e:
        print(f"\n⚠️ 응답 파싱 실패: {e}")
        print(f"   - 전체 응답: {response_data}")
        return None

def process_video(video_path: str, time_interval: int, question: str):
    """
    비디오 프레임을 LLM API로 분석하고, 답변을 자막처럼 영상에 그려넣습니다.
    (Base64 인코딩을 사용하여 임시 파일 없이 처리합니다.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: 비디오 파일 '{video_path}'을(를) 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = f"{os.path.splitext(video_path)[0]}_subtitled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    print(f"🚀 LLM 비디오 분석을 시작합니다 (Base64 모드): {video_path}")
    print(f"   - 질문: '{question}'")
    print(f"   - {time_interval} 프레임마다 분석을 수행합니다.")
    print(f"   - 결과는 '{output_filename}' 파일에 저장됩니다.")

    latest_subtitle = None

    for frame_index in tqdm(range(total_frames), desc="처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % time_interval == 0:
            # --- 변경점: 임시 파일 대신 Base64 인코딩으로 직접 변환 ---
            # 1. OpenCV 프레임(Numpy 배열)을 JPEG 형식으로 메모리 내에서 인코딩
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print(f"\n⚠️ 프레임 {frame_index}: 이미지 인코딩 실패")
                continue

            # 2. 인코딩된 데이터를 Base64 문자열로 변환하고 Data URI 생성
            base64_string = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_string}"

            # 3. API 요청 데이터 생성
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<image>\n{question}"},
                        # file:// 경로 대신 data URI 사용
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }],
                "max_tokens": MAX_TOKENS
            }

            # 4. API 요청 (try/except 블록은 그대로 유지)
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    subtitle = parse_response(response.json())
                    if subtitle:
                        latest_subtitle = subtitle
                else:
                    print(f"\n⚠️ 프레임 {frame_index}: API 오류 (상태 코드: {response.status_code}, 내용: {response.text})")
            except requests.exceptions.RequestException as e:
                print(f"\n❌ 프레임 {frame_index}: API 연결 오류 - {e}")
            # --- 변경 완료 ---

        # 5. 현재 프레임에 마지막 자막 그리기 (이 부분은 동일)
        if latest_subtitle:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (255, 255, 255) # 흰색
            thickness = 2
            
            text_size = cv2.getTextSize(latest_subtitle, font, font_scale, thickness)[0]
            position = ((width - text_size[0]) // 2, height - 40)

            bg_pos_start = (position[0] - 10, position[1] - text_size[1] - 5)
            bg_pos_end = (position[0] + text_size[0] + 10, position[1] + 10)
            cv2.rectangle(frame, bg_pos_start, bg_pos_end, (0, 0, 0), -1)
            
            cv2.putText(frame, latest_subtitle, position, font, font_scale, color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 처리 완료! 결과가 '{output_filename}'에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM을 이용해 비디오를 분석하고 결과를 자막으로 합성합니다.")
    parser.add_argument("video_path", type=str, help="처리할 원본 비디오 파일의 경로")
    parser.add_argument("question", type=str, help="각 프레임에 대해 질문할 내용")
    parser.add_argument("--time_interval", type=int, default=150, help="분석을 수행할 프레임 간격 (기본값: 150)")
    
    args = parser.parse_args()
    
    process_video(args.video_path, args.time_interval, args.question)