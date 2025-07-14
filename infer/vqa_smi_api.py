import cv2
import requests
import base64
import argparse
import os
from tqdm import tqdm
from moviepy import VideoFileClip, TextClip, CompositeVideoClip

# --- 설정 ---
API_URL = "http://127.0.0.1:9997/v1/chat/completions"
MODEL_NAME = "InternVL3"
MAX_TOKENS = 256

def parse_response(response_data: dict) -> str | None:
    """API 응답(JSON)을 파싱하여 모델의 답변 텍스트를 추출합니다."""
    try:
        content = response_data["choices"][0]["message"]["content"]
        return content.strip()
    except (KeyError, IndexError, TypeError) as e:
        print(f"\n⚠️ 응답 파싱 실패: {e}")
        return None

def collect_subtitles_from_video(video_path: str, time_interval: int, question: str, fps: float) -> list:
    """(1단계) 비디오를 스캔하며 API로부터 자막 데이터를 수집합니다."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    subtitles = []
    last_subtitle_text = None
    
    print("(1/2) 🚀 API로 자막 데이터를 수집합니다...")
    for frame_index in tqdm(range(total_frames), desc="자막 수집 중"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % time_interval == 0:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue

            base64_string = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{base64_string}"

            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"<image>\n{question}"},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]}],
                "max_tokens": MAX_TOKENS
            }

            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    new_subtitle = parse_response(response.json())
                    if new_subtitle and new_subtitle != last_subtitle_text:
                        current_time = frame_index / fps
                        
                        # 이전 자막이 있었다면 종료 시간 업데이트
                        if subtitles:
                            subtitles[-1]['end'] = current_time
                        
                        # 새 자막 추가
                        subtitles.append({'start': current_time, 'end': total_frames / fps, 'text': new_subtitle})
                        last_subtitle_text = new_subtitle
            except requests.exceptions.RequestException as e:
                print(f"\n❌ 프레임 {frame_index}: API 연결 오류 - {e}")
    
    cap.release()
    return subtitles

def create_video_with_subtitles(video_path: str, subtitles: list, output_filename: str):
    """(2단계) Moviepy를 사용해 원본 영상에 자막을 합성합니다."""
    print("\n(2/2) 🎬 Moviepy로 비디오를 합성합니다... (시간이 걸릴 수 있습니다)")
    
    video_clip = VideoFileClip(video_path)
    text_clips = []

    for sub in subtitles:
        # TextClip 생성 (자동 줄바꿈을 위해 method='caption' 사용)
        width = int(video_clip.w * 0.8)
        txt_clip = TextClip(
            text = sub['text'],
            font_size=30,
            color='white',
            font="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/arial.ttf",
            bg_color=(0, 0, 0, 153),
            size= (width, None),
            method='caption'
        ).with_position(('center', 0.8), relative=True).with_start(sub['start']).with_duration(sub['end'] - sub['start'])  
        
        text_clips.append(txt_clip)

    # 원본 비디오와 자막 클립들을 합성
    final_clip = CompositeVideoClip([video_clip] + text_clips)
    
    # 최종 비디오 파일로 저장 (오디오 포함)
    final_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac')
    video_clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM을 이용해 비디오를 분석하고 결과를 Moviepy 자막으로 합성합니다.")
    parser.add_argument("video_path", type=str, help="처리할 원본 비디오 파일의 경로")
    parser.add_argument("question", type=str, help="각 프레임에 대해 질문할 내용")
    parser.add_argument("--time_interval", type=int, default=150, help="분석을 수행할 프레임 간격 (기본값: 150)")
    
    args = parser.parse_args()
    
    # 영상 FPS 정보 미리 가져오기
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        print(f"❌ Error: 비디오 파일 '{args.video_path}'의 FPS 정보를 읽을 수 없습니다.")
    else:
        # 1단계: 자막 데이터 수집
        subtitles_data = collect_subtitles_from_video(args.video_path, args.time_interval, args.question, fps)
        
        if subtitles_data:
            # 2단계: 영상 합성
            output_file = f"{os.path.splitext(args.video_path)[0]}_subtitled_moviepy.mp4"
            create_video_with_subtitles(args.video_path, subtitles_data, output_file)
            print(f"\n✅ 처리 완료! 결과가 '{output_file}'에 저장되었습니다.")
        else:
            print("\n- 생성된 자막이 없어 비디오 합성을 건너뜁니다.")