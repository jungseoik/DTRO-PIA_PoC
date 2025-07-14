import streamlit as st
from ui.component.ui_video_input import local_input_vqa_video
import cv2
import os
import threading
from queue import Queue
import queue
import time
from ui.component.ui_alarm_count import render_count_status_ui
from ui.component.ui_progress import run_progress_bar_vqa
from decord import VideoReader, cpu
from env.config import PROMPT_V3 , PROMPT_V2
from utils.api.vqa_api import internvl_vision_api_response_vqa
from env.config import MAX_HEIGHT, MAX_WIDTH, API_URL
from utils.clip_ebc_onnx import ClipEBCOnnx
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import LetterBox
transform = LetterBox(new_shape=(MAX_HEIGHT, MAX_WIDTH), scaleup=False)

def render_description_section_vqa():
    """스타일리시한 Description 섹션 렌더링"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h3 style="color: white; margin: 0; font-weight: 600;">📝 Description</h3>
    </div>
    """, unsafe_allow_html=True)
    return st.empty()

def render_alarm_section_vqa():
    """스타일리시한 Alarm 섹션 렌더링"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <h3 style="color: white; margin: 0; font-weight: 600;">🚨 Alarm Status</h3>
    </div>
    """, unsafe_allow_html=True)
    return st.empty()

def display_description_result_vqa(container, desc_result):
    """Description 결과를 스타일리시하게 출력"""
    container.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 25px; border-radius: 15px; 
                border: 2px solid #667eea; 
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.15);
                margin-top: -10px; margin-bottom: 20px;">
        <p style="font-size: 18px; font-weight: 600; line-height: 1.7; 
                  color: #2c3e50; margin: 0; text-align: justify;
                  text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            {desc_result}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_alarm_result_vqa(container, category_result):
    """Alarm 결과를 상태에 따라 스타일리시하게 출력"""
    if category_result.lower() == "normal":
        alarm_style = """
        <div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
            <span style="color: white; font-size: 18px; font-weight: 600;">
                ✅ {status}
            </span>
        </div>
        """
    else:
        alarm_style = """
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    animation: pulse 2s infinite;">
            <span style="color: white; font-size: 18px; font-weight: 600;">
                ⚠️ {status}
            </span>
        </div>
        <style>
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
        </style>
        """
    
    container.markdown(alarm_style.format(status=category_result), unsafe_allow_html=True)


class InferenceConsumer_vqa(threading.Thread):
    def __init__(
        self,
        video_name: str,
        frame_queue_infer: Queue,
        frame_queue_result: Queue,
    ):
        super().__init__()
        self.video_name = video_name
        self.frame_queue_infer = frame_queue_infer
        self.frame_queue_result = frame_queue_result
        self.running = True


    def run(self):
        while self.running:
            try:
                item = self.frame_queue_infer.get(timeout=1)
                frame = item["frame"]
                frame_idx = item["frame_idx"]
                is_last = item["is_last"]

                print(f"🧠 Inference: frame_idx={frame_idx}, is_last={is_last}")
                result_cate, result_dsec = internvl_vision_api_response_vqa(frame, question = PROMPT_V2)
                if result_cate:
                    self.frame_queue_result.put((frame_idx, result_cate, result_dsec), timeout=1)
                
                if is_last:
                    print("🛑 InferenceConsumer: 마지막 프레임 처리 완료")
                    break

            except queue.Empty:
                print("⏳ InferenceConsumer: 대기 중...")
                continue
            except queue.Full:
                print("⚠️ 결과 큐가 가득 찼습니다.")
                time.sleep(0.2)


class FrameProducer_vqa(threading.Thread):
    def __init__(
        self, 
        video_path: str, 
        frame_queue: Queue, 
        frame_queue_infer: Queue, 
        time_interval: int, 
        target_fps: int = 30
    ):
        super().__init__()
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.frame_queue_infer = frame_queue_infer
        self.time_interval = time_interval
        self.target_fps = target_fps
        self.is_running = True
        self.infer_q_size = None

    def run(self):
        print("⚡ decord VideoReader 초기화 중...")
        vr = VideoReader(self.video_path, ctx=cpu(0))
        total_frames = len(vr)
        print(f"------------총 프레임 수: {total_frames}")
        infer_indices = list(range(0, total_frames, self.time_interval))
        print(f"------------추론 인덱스: {len(infer_indices)}")
        self.infer_q_size = len(infer_indices)
        # 추론 프레임 먼저 decord로 빠르게 가져와서 큐에 넣기
        infer_frames = vr.get_batch(infer_indices).asnumpy()  # (N, H, W, 3)
        # 이거 맞냐?
        infer_frames = infer_frames[:, :, :, ::-1]

        for i, frame_idx in enumerate(infer_indices):
            frame = infer_frames[i]
            is_last = frame_idx == total_frames - 1
            item = {
                "frame": frame,
                "frame_idx": frame_idx,
                "is_last": is_last,
                "model_input": True
            }

            while True:
                try:
                    self.frame_queue_infer.put_nowait(item)
                    print(f"📤 Infer Put: frame_idx={frame_idx}, qsize={self.frame_queue_infer.qsize()}")
                    break
                except Exception:
                    print("⏳ frame_queue_infer 가득 참. 대기 중...")
                    time.sleep(0.01)

        # 2전체 프레임 순차적으로 frame_queue에 전송 (fps 제한)
        delay = 1.0 / self.target_fps
        for frame_idx in range(total_frames):
            frame = vr[frame_idx].asnumpy()
            frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
            is_last = frame_idx == total_frames - 1
            item = {
                "frame": frame,
                "frame_idx": frame_idx,
                "is_last": is_last,
                "model_input": frame_idx in infer_indices
            }

            while True:
                try:
                    self.frame_queue.put(item, timeout=1)
                    print(f"🟢 Put frame_idx={frame_idx}, is_last={is_last}, qsize={self.frame_queue.qsize()}")
                    break
                except Exception:
                    print("⏳ frame_queue 가득 참. 대기 중...")
                    time.sleep(0.05)

            time.sleep(delay)

            if is_last:
                break

        self.is_running = False
        print("🏁 FrameProducer 종료")


def render_from_queues_interval_vqa(
    frame_queue,
    infer_result_queue,
    frame_origin,
    description_output_ui,
    alarm_output_ui,
    time_interval,
):

    while True:
        if frame_queue.empty():
            print("🛑 frame_queue 비었음. 종료.")
            st.success("✅ 처리 완료!")
            break

        try:
            item = frame_queue.get_nowait()
        except queue.Empty:
            continue

        frame = item["frame"]
        frame_idx = item["frame_idx"]
        is_last = item["is_last"]

        # 항상 출력: 원본 프레임
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = transform(image=frame_rgb)
        frame_origin.image(frame_rgb, use_container_width=True)
        print(f"🎞️ frame_origin 출력: {frame_idx}")

        # 조건 만족 시: 결과 큐에서 동기 pop
        if frame_idx % time_interval == 0:
            try:
                idx_c, cate, desc = infer_result_queue.get_nowait()
            except queue.Empty:
                print(f"⚠️ 결과 큐 비어 있음 @frame_idx={frame_idx}")
                continue
        
            if not (frame_idx == idx_c):
                print(f"❌ 인덱스 불일치: frame={frame_idx}, count={idx_c}")
            else:
                display_description_result_vqa(description_output_ui, desc)
                display_alarm_result_vqa(alarm_output_ui, cate)

        time.sleep(1.0 / time_interval)

        if is_last:
            print("✅ 마지막 프레임 도달. 종료.")
            st.success("✅ 처리 완료!")
            break



def vqa_video_tab():

    col_video_vqa, col_people_count = st.columns(2, gap="medium")

    with col_video_vqa:
        st.subheader("Video Input")
        local_input_vqa_video()

    button_vaq =  st.button("▶️ VQA Inference")
    st.divider()

    col_original, col_people_count= st.columns(2, gap="medium")
    with col_original:
        st.subheader("Video")
        st.session_state.video_falldown_output = st.empty()
    with col_people_count:
        st.subheader("Output")
        st.session_state.video_falldown_description_output = render_description_section_vqa()
        st.session_state.video_falldown_alarm_output = render_alarm_section_vqa()


    if button_vaq:
        if not st.session_state.video_falldown_path:
            st.warning("먼저 비디오를 업로드 해주세요.")
        else:
            
            producer = FrameProducer_vqa(
                video_path=st.session_state.video_falldown_path,
                frame_queue=st.session_state.frame_queue_vqa,
                frame_queue_infer=st.session_state.frame_queue_infer_vqa,
                time_interval=st.session_state.time_interval
            )

            producer.start()
            inference_thread = InferenceConsumer_vqa(
                video_name=os.path.basename(st.session_state.video_falldown_path),
                frame_queue_infer=st.session_state.frame_queue_infer_vqa,
                frame_queue_result=st.session_state.frame_queue_infer_result_vqa,
            )
            inference_thread.start()

            run_progress_bar_vqa(st.session_state.progress_duration_vqa)
            render_from_queues_interval_vqa(
                frame_queue=st.session_state.frame_queue_vqa,
                infer_result_queue = st.session_state.frame_queue_infer_result_vqa,
                frame_origin=st.session_state.video_falldown_output,
                description_output_ui=st.session_state.video_falldown_description_output,
                alarm_output_ui= st.session_state.video_falldown_alarm_output,
                time_interval=st.session_state.time_interval,
            )    




