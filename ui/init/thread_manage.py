import gc
import threading
from collections import deque
from queue import Queue
import numpy as np
import streamlit as st
import torch

class ThreadManager:
    def __init__(self):
        self.thread_enabled = False
        self.rtsp_frame_queue = Queue(maxsize=100)
        self.sim_scores_queue = Queue(maxsize=100)

        self.frame_queue: deque = deque()
        self.frame_queue_updated = False
        self.text_vectors: np.ndarray = None
        self.out_sim_scores = None
        self.out_sim_softmax = None
        self.thread_lock = threading.Lock()
        self.thread: threading.Thread = None

        self.select_video_output_method = None

        # 이벤트 플래그를 클래스 속성으로 추가
        self.frame_queue_updated_event = threading.Event()  # 프레임 큐 업데이트 이벤트
        self.model_processing_done_event = threading.Event()  # 모델 처리 완료 이벤트

        self.category_info = None
        self.normal_index = None


class RunParameters:
    def __init__(
        self,
        video_path: str,
        window_size: int,
        video_method: str,
        model,
        time_sampling,
        text_vectors: np.ndarray,
        cfg_type,
        roi_ds: None,
        qsize=300,
    ):
        self.video_path = video_path
        self.video_method = video_method

        # video info
        self.fps: int = None
        self.video_width: int = None
        self.video_height: int = None
        self.total_frame: int = None

        self.rtsp_frame_queue = Queue(qsize)
        self.model_inference_queue = Queue(qsize)

        self.window_size = window_size
        self.time_sampling = time_sampling
        self.text_vectors: np.ndarray = text_vectors
        self.out_sim_scores = None
        self.out_sim_softmax = None
        self.model = model
        self.cfg_type = cfg_type
        self.roi_ds = roi_ds

        # for control the threads
        self.isrunning = True


def wait_thread(thread_name: str):
    for thread in threading.enumerate():
        if thread.name == thread_name:
            thread.join()
