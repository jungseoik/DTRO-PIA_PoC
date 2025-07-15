import cv2
import requests
import base64
import os
from tqdm import tqdm
from utils.clip_ebc_onnx import ClipEBCOnnx
import pandas as pd
import glob
import shutil
API_URL = "http://localhost:8000/predict_json" 

def process_video_ebc2(video_path: str, time_interval: int):
    """
    비디오를 프레임 단위로 읽어 커스텀 모델로 추론하고,
    결과를 리눅스 /tmp 경로에 저장합니다 (원본, 처리된 영상, CSV).
    """
    # 경로 설정
    tmp_dir = "/tmp/ebc_video_result"
    os.makedirs(tmp_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    tmp_video_path = os.path.join(tmp_dir, f"{base_filename}_original.mp4")
    output_filename = os.path.join(tmp_dir, f"{base_filename}_processed.mp4")
    csv_filename = os.path.join(tmp_dir, f"{base_filename}_ebc_video_result.csv")

    # 입력 비디오 /tmp 복사
    shutil.copy(video_path, tmp_video_path)

    # 비디오 열기
    cap = cv2.VideoCapture(tmp_video_path)
    if not cap.isOpened():
        print(f"❌ Error: 비디오 파일 '{tmp_video_path}'을(를) 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    csv_data = []  # 결과 저장 리스트

    # 모델 초기화
    model = ClipEBCOnnx(onnx_model_path="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/CLIP_EBC_nwpu_rmse_onnx.onnx")

    print(f"🚀 비디오 처리 시작: {tmp_video_path}")
    print(f"   - 총 프레임: {total_frames}, FPS: {fps:.2f}")
    print(f"   - {time_interval} 프레임마다 추론 수행")
    print(f"   - 처리 영상: {output_filename}")
    print(f"   - CSV 저장: {csv_filename}")
    print(f"   - 원본 복사본: {tmp_video_path}")

    latest_text = None

    for frame_index in tqdm(range(total_frames), desc="처리 중"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % time_interval == 0:
            try:
                count = model.predict(frame)
                time_seconds = frame_index / fps
                minutes = int(time_seconds // 60)
                seconds = time_seconds % 60
                time_str = f"{minutes:02d}:{seconds:05.2f}"

                csv_data.append({
                    'frame': frame_index,
                    'time': time_str,
                    'count': count
                })

                # 저장용 파일명 생성
                frame_filename = os.path.join(
                    tmp_dir,
                    f"{base_filename}_frame{frame_index:05d}_count{count:.2f}.jpg"
                )
                # 이미지 저장
                cv2.imwrite(frame_filename, frame)

                latest_text = f"Count: {count:.2f}"
            except Exception as e:
                print(f"\n❌ 프레임 {frame_index}: 추론 오류 - {e}")

        if latest_text:
            position = (width - 300, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            color = (0, 255, 0)
            thickness = 2
            cv2.putText(frame, latest_text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"📊 CSV 저장 완료: '{csv_filename}' ({len(csv_data)}개 항목)")

    print(f"\n✅ 처리 완료! 결과가 /tmp 에 저장되었습니다.")

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


def process_image_ebc(folder_path: str):
    """
    폴더 내 모든 이미지 파일을 처리하여 추론 결과를 CSV로 저장합니다.
    - 추론에 사용된 이미지들을 /tmp/ebc_image_results/ 경로에 복사
    - 추론 결과 CSV는 원본 폴더 및 /tmp에 각각 저장
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: 폴더 '{folder_path}'을(를) 찾을 수 없습니다.")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_files:
        print(f"❌ Error: 폴더 '{folder_path}'에서 이미지 파일을 찾을 수 없습니다.")
        return

    # 모델 초기화
    model = ClipEBCOnnx(onnx_model_path="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/CLIP_EBC_nwpu_rmse_onnx.onnx")

    # /tmp 경로 설정
    tmp_dir = "/tmp/ebc_image_results"
    os.makedirs(tmp_dir, exist_ok=True)

    # CSV 파일 경로
    csv_filename_original = os.path.join(folder_path, "ebc_image_results.csv")
    csv_filename_tmp = os.path.join(tmp_dir, "ebc_image_results.csv")

    print(f"🚀 이미지 추론 시작: {folder_path}")
    print(f"   - 총 이미지: {len(image_files)}개")
    print(f"   - 원본 CSV 저장: {csv_filename_original}")
    print(f"   - /tmp CSV 저장: {csv_filename_tmp}")
    print(f"   - 이미지 사본 저장 디렉토리: {tmp_dir}")

    results = []

    for image_path in tqdm(image_files, desc="이미지 처리 중"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"\n⚠️ 이미지를 읽을 수 없습니다: {image_path}")
                continue

            count = model.predict(image)
            filename = os.path.basename(image_path)

            # 이미지 복사 (/tmp)
            shutil.copy(image_path, os.path.join(tmp_dir, filename))

            results.append({
                'filename': filename,
                'count': count
            })

        except Exception as e:
            print(f"\n❌ 이미지 처리 오류 ({os.path.basename(image_path)}): {e}")

    if results:
        df = pd.DataFrame(results)
        # 원래 위치에 저장
        df.to_csv(csv_filename_original, index=False, encoding='utf-8')
        print(f"📁 원본 폴더 CSV 저장 완료: {csv_filename_original}")
        
        # /tmp 경로에도 저장
        df.to_csv(csv_filename_tmp, index=False, encoding='utf-8')
        print(f"📁 /tmp 경로 CSV 저장 완료: {csv_filename_tmp}")
        
        print(f"✅ 모든 이미지 처리 및 결과 저장 완료!")
    else:
        print("\n❌ 처리된 이미지가 없습니다.")


def process_image_ebc_dtro(folder_path: str, save_dot_map: bool = True):
    """
    이미지 추론 + 시각화 결과를 assets 및 /tmp에 모두 저장하는 버전.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: 폴더 '{folder_path}'을(를) 찾을 수 없습니다.")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']

    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_files:
        print(f"❌ Error: 폴더 '{folder_path}'에서 이미지 파일을 찾을 수 없습니다.")
        return

    model = ClipEBCOnnx(onnx_model_path="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/CLIP_EBC_nwpu_rmse_onnx.onnx")

    # 저장 디렉토리
    tmp_dir = "/tmp/ebc_image_det_dense_results"
    asset_dir = "assets/ebc_image_results"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(asset_dir, exist_ok=True)

    # CSV 파일
    csv_filename_original = os.path.join(folder_path, "ebc_image_results.csv")
    csv_filename_tmp = os.path.join(tmp_dir, "ebc_image_results.csv")

    results = []

    print(f"🚀 DTRO 이미지 추론 시작 - 총 {len(image_files)}개")

    for image_path in tqdm(image_files, desc="이미지 처리 중"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"\n⚠️ 이미지를 읽을 수 없습니다: {image_path}")
                continue

            filename = os.path.basename(image_path)
            name_only, _ = os.path.splitext(filename)

            # 예측
            count = model.predict(image)

            # 시각화
            vis1_fig, vis1_img = model.visualize_density_map(save=True, save_path=os.path.join(asset_dir, f"{name_only}_density.png"))
            shutil.copy(os.path.join(asset_dir, f"{name_only}_density.png"), os.path.join(tmp_dir, f"{name_only}_density.png"))

            # Dot map 저장 여부 옵션
            if save_dot_map:
                vis2_fig, vis2_img = model.visualize_dots(
                    save=True,
                    save_path=os.path.join(asset_dir, f"{name_only}_dots.png")
                )
                if vis2_img is not None:
                    shutil.copy(
                        os.path.join(asset_dir, f"{name_only}_dots.png"),
                        os.path.join(tmp_dir, f"{name_only}_dots.png")
                    )
            # 원본 이미지도 /tmp에 복사
            shutil.copy(image_path, os.path.join(tmp_dir, filename))

            # 결과 저장
            results.append({
                'filename': filename,
                'count': count
            })

        except Exception as e:
            print(f"\n❌ 오류 - {os.path.basename(image_path)}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_filename_original, index=False, encoding='utf-8')
        df.to_csv(csv_filename_tmp, index=False, encoding='utf-8')

        print(f"\n📁 CSV 저장 완료: {csv_filename_original}, {csv_filename_tmp}")
        print(f"🖼️ 이미지 및 시각화 결과 저장 완료: {asset_dir}, {tmp_dir}")
        print(f"✅ DTRO 이미지 추론 및 저장 완료")
    else:
        print("\n❌ 처리된 이미지가 없습니다.")