import threading
import time

import cv2
import numpy as np
import streamlit as st
from collections import deque, defaultdict
import math 
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import LetterBox

from config import (
    MAX_HEIGHT,
    MAX_WIDTH,
    RTSP_REDIRECTION_CNT,
    RTSP_TARGET_FPS,
    UI_DRAW_DELAY_BIAS,
    GRAPH_MAX_LEN,
    DEFAULT_USER_DATA_SAVE_DIR,
)
from manage.thread_manage import RunParameters
from ui.component.graph import update_graph, append_prompts_queue
from ui.component.text_output import make_text_output
from datetime import datetime
transform = LetterBox(new_shape=(MAX_HEIGHT, MAX_WIDTH), scaleup=False)

class VideoReadThread(threading.Thread):
    def __init__(self, name, args) -> None:
        super().__init__(name=name, args=args)
        self.is_running = True
        self.args = args[0] 

    def run(self):
        global RTSP_TARGET_FPS
        vid = cv2.VideoCapture(self.args.video_path)
        fps = int(round(vid.get(cv2.CAP_PROP_FPS),0))
        self.args.fps = fps
        self.args.video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.args.video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.args.total_frame = (
            int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
            else ""
        )
        if fps > 999 or fps < 1:
            print(f"fps have problem (now : {fps}) so change to 30")
            self.args.fps = 30
        else:
            self.args.fps = fps

        # UI draw에서 30fps로 그리기에는 workstation의 성능이 받쳐줄 수 없어서 설정된 프레임만 받도록 수정
        analysis_interval = self.args.time_sampling
        # 최소공배수를 계산하여 최소공배수가 되면 cnt를 초기화 하기 위해서 
        lcm_period, analysis_index, display_interval = calc_frame_display_interval(fps, analysis_interval=analysis_interval, display_fps=RTSP_TARGET_FPS)
        

        cnt = 0
        rtsp_redirection_cnt = 0
        while self.is_running:
            ret, frame = vid.read()
            if ret:
                cnt += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cnt in analysis_index :
                    self.args.rtsp_frame_queue.put([frame, True])  # 분석을 해야하는 프레임
                elif cnt in display_interval :
                    self.args.rtsp_frame_queue.put([frame, False])  # 분석을 하면 안되는 프레임
                if lcm_period == cnt : cnt = 0 # 최소공배수가 되면 cnt 초기화
            else:
                if self.args.video_method == "image" or self.args.video_method == "video":
                    break
                else:
                    print("RTSP have problem, it provide return false")
                    rtsp_redirection_cnt += 1
                    if rtsp_redirection_cnt > RTSP_REDIRECTION_CNT * fps:
                        vid = cv2.VideoCapture(self.args.video_path)
                        if vid.isOpened():
                            rtsp_redirection_cnt = 0
                            print("RTSP has been opened")
                        else:
                            print("Can't open RTSP stream")


def calc_frame_display_interval(fps, analysis_interval, display_fps):
    display_interval = int(round(fps / display_fps,0))
    lcm_period = abs(analysis_interval * display_interval) // math.gcd(analysis_interval, display_interval)
    analysis_index = list(range(0,lcm_period+1, analysis_interval))[1:]
    display_index = list(range(0, lcm_period+1, display_interval))[1:]
    display_interval = sorted(set(display_index) - set(analysis_index))
    return lcm_period, analysis_index, display_interval

def UI_draw(params: RunParameters):
    counter = 0
    sim_score_queue = defaultdict(MyDeque)
    # TODO : sim_score를 파일로 저장하여 나중에 필요한 곳에서 불러와서 동작할 수 있도록 sim score 저장필요
    # save_file_path = os.path.join(DEFAULT_USER_DATA_SAVE_DIR, f"{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{st.session_state.user_id}.csv")
    ######## st.session_state.user_id = '2deab3b9-d8cc-4c62-b8bf-8c534fd5ce2a' << 해당 값과 date
    while params.isrunning:
        data = params.model_inference_queue.get()
        sim_scores = data["sim_scores"]
        frame = data["frame"]
        t1 = time.time()
        update_video_show(frame)

        if sim_scores is not None:
            append_prompts_queue(sim_score_queue, sim_scores)
            update_graph(sim_score_queue)
            make_text_output(sim_scores, cfg_type=params.cfg_type)

        st.session_state.ui_frame_number.metric(
            label="Total Sales", value=f"{int(counter)}/ {params.total_frame}"
        )
        counter += (params.fps / RTSP_TARGET_FPS)

        UI_draw_spend_time = time.time() - t1
        sleep_timer = (1 / RTSP_TARGET_FPS) - UI_draw_spend_time - UI_DRAW_DELAY_BIAS
        if sleep_timer < 0:
            sleep_timer = 0

        time.sleep(sleep_timer)  # FPS가 30일 때 ui를 그리는데 걸리는 시간을 뺴서 더 조금 sleep하도록
        
        # If video is over, break
        if counter >= params.total_frame - (params.fps / RTSP_TARGET_FPS):
            st.session_state.prompt_score_dict = sim_score_queue
            break

def update_video_show(frame):
    source_frame = st.session_state.video_output_frame
    frame = transform(image=frame)
    source_frame.image(frame, use_container_width=True)

class MyDeque(deque):
    DEFAULT_MAXLEN = GRAPH_MAX_LEN
    def __init__(self, iterable=()):
        super().__init__(iterable, maxlen=self.DEFAULT_MAXLEN)
