import streamlit as st
import time

# def run_progress_bar(seconds: int = 10):
#     progress = st.progress(0)
#     steps = 100
#     for i in range(steps):
#         time.sleep(seconds / steps)
#         progress.progress(i + 1)


def run_progress_bar(seconds: int = 10):
    progress = st.progress(0)
    steps = 100
    for i in range(steps):
        time.sleep(10 / steps)
        progress.progress(i + 1)

def run_progress_bar_vqa(seconds: int = 10):
    progress = st.progress(0)
    steps = 100
    for i in range(steps):
        time.sleep(10 / steps)
        progress.progress(i + 1)
        
    # total = st.session_state.infer_q_size
    # progress = st.progress(0)
    # while True:
    #     try:
    #         qsize = st.session_state.frame_queue_infer.qsize()
    #     except:
    #         qsize = 0

    #     completed = total - qsize
    #     percent = int((completed / total) * 100)
    #     progress.progress(min(percent, 100))

    #     if qsize <= 0:
    #         break

    #     time.sleep(0.1)  # 너무 빠르게 돌지 않도록 제한