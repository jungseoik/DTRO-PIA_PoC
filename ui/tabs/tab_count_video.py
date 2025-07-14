import streamlit as st
from ui.component.ui_video_input import local_input_video
import cv2
import os
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import LetterBox
import threading
from queue import Queue
import queue
import time
from ui.component.ui_alarm_count import render_count_status_ui
from ui.component.ui_progress import run_progress_bar
from ui.component.ui_sidebar import siderbar_setting_ui
from env.config import MAX_HEIGHT, MAX_WIDTH, API_URL
from decord import VideoReader, cpu
transform = LetterBox(new_shape=(MAX_HEIGHT, MAX_WIDTH), scaleup=False)

class InferenceConsumer(threading.Thread):
    def __init__(
        self,
        video_name: str,
        frame_queue_infer: Queue,
        frame_queue_count_result: Queue,
        model
    ):
        super().__init__()
        self.video_name = video_name
        self.frame_queue_infer = frame_queue_infer
        self.frame_queue_count_result = frame_queue_count_result
        self.running = True
        self.model = model

    def run(self):
        while self.running:
            try:
                item = self.frame_queue_infer.get(timeout=1)
                frame = item["frame"]
                frame_idx = item["frame_idx"]
                is_last = item["is_last"]

                print(f"🧠 Inference: frame_idx={frame_idx}, is_last={is_last}")
                result = self.model.predict(frame)
                if result:
                    self.frame_queue_count_result.put((frame_idx, result), timeout=1)

                if is_last:
                    print("🛑 InferenceConsumer: 마지막 프레임 처리 완료")
                    break

            except queue.Empty:
                print("⏳ InferenceConsumer: 대기 중...")
                continue
            except queue.Full:
                print("⚠️ 결과 큐가 가득 찼습니다.")
                time.sleep(0.2)


class FrameProducer(threading.Thread):
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


def render_from_queues_interval(
    frame_queue,
    count_result_queue,
    frame_origin,
    count_ph,
    transform,
    time_interval,
):
    import queue

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
                idx_c, count = count_result_queue.get_nowait()
            except queue.Empty:
                print(f"⚠️ 결과 큐 비어 있음 @frame_idx={frame_idx}")
                continue
            
            # 인덱스 확인
            if not (frame_idx == idx_c):
                print(f"❌ 인덱스 불일치: frame={frame_idx}, count={idx_c}")
            else:
                render_count_status_ui(count, frame_idx, count_ph)

        time.sleep(1.0 / time_interval)

        if is_last:
            print("✅ 마지막 프레임 도달. 종료.")
            st.success("✅ 처리 완료!")
            break



def count_video_tab():

    col_video, col_people_count = st.columns(2, gap="medium")

    with col_video:
        st.subheader("Video Input")
        local_input_video()
        siderbar_setting_ui()

    button =  st.button("▶️ Run Inference")
    st.divider()

    col_original, col_people_count= st.columns(2, gap="medium")
    with col_original:
        st.subheader("Video")
        st.session_state.origin = st.empty()
    with col_people_count:
        st.subheader("People Count")
        st.session_state.count_ui = st.empty()

    if button:
        if not st.session_state.video_path:
            st.warning("먼저 비디오를 업로드 해주세요.")
        else:

            vr = VideoReader(st.session_state.video_path, ctx=cpu(0))
            total_frames = len(vr)
            infer_q_size = len(range(0, total_frames, st.session_state.time_interval))
            st.session_state.infer_q_size = infer_q_size

            producer = FrameProducer(
                video_path=st.session_state.video_path,
                frame_queue=st.session_state.frame_queue,
                frame_queue_infer=st.session_state.frame_queue_infer,
                time_interval=st.session_state.time_interval
            )

            producer.start()
            inference_thread = InferenceConsumer(
                video_name=os.path.basename(st.session_state.video_path),
                frame_queue_infer=st.session_state.frame_queue_infer,
                frame_queue_count_result=st.session_state.frame_queue_count_result,
                model=st.session_state.clip_ebc_model
            )
            inference_thread.start()

            run_progress_bar(st.session_state.progress_duration)
            render_from_queues_interval(
                frame_queue=st.session_state.frame_queue,
                count_result_queue=st.session_state.frame_queue_count_result,
                frame_origin=st.session_state.origin,
                count_ph=st.session_state.count_ui,
                transform=transform,
                time_interval=st.session_state.time_interval,
            )    




