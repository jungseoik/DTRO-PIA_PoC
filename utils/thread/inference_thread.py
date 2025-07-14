import threading
import time
from collections import deque
from typing import Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from utils.draw_roi import crop_with_polyline, draw_polyline


def inference_model(frames: torch.Tensor, model) -> Tuple[np.ndarray, np.ndarray]:
    """
    모델을 실행하여 프레임 벡터와 텍스트 벡터 간 유사도를 계산합니다.

    Args:
        frames (torch.Tensor): 처리할 프레임 텐서
        thread_manager (ThreadManager): 상태 관리를 담당하는 ThreadManager 인스턴스

    Returns:
        Tuple[np.ndarray, np.ndarray]: 유사도 점수 배열과 소프트맥스 결과 배열
    """
    sim_scores = model.get_online_simScore(frames)
    max_values, max_indices = sim_scores.max(dim=1)
    softmax_values = F.softmax(max_values, dim=0)

    sim_scores_np = max_values.cpu().numpy()
    softmax_values_np = softmax_values.cpu().numpy()

    # NumPy 배열로 변환된 값 리턴
    return sim_scores_np, softmax_values_np


class InferenceThread(threading.Thread):
    def __init__(self, name, args) -> None:
        super().__init__(name=name, args=args)
        self.is_running = True
        self.args = args[0]

    def run(self) -> None:
        window_size_deque = deque(maxlen=self.args.window_size)
        while self.is_running:
            frame, state = self.args.rtsp_frame_queue.get()
        
            displayed_frame = (
                draw_polyline(frame, self.args.roi_ds).copy() if self.args.roi_ds else frame
            )
            if not state:  # state가 false 면 그냥 프레임만 넘김
                self.args.model_inference_queue.put(
                    {"sim_scores": None, "sim_softmax": None, "frame": displayed_frame}
                )

            else:  # state 가 있으면 분석 하고 sim도 같이 넘김
                cropped_frame = (
                    crop_with_polyline(frame, self.args.roi_ds).copy() if self.args.roi_ds else frame
                )
                window_size_deque.append(cropped_frame)
                # 설정한 window size 만큼 큐를 초기에 채움
                while len(window_size_deque) != window_size_deque.maxlen:
                    window_size_deque.append(cropped_frame)
                frames = np.array(window_size_deque)
                sim_scores, sim_softmax = inference_model(frames, self.args.model)
                # inference 결과를 queue에 쌓음
                self.args.model_inference_queue.put(
                    {"sim_scores": sim_scores, "sim_softmax": sim_softmax, "frame": displayed_frame}
                )
