import streamlit as st
from queue import Queue
from utils.clip_ebc_onnx import ClipEBCOnnx

def declare_all_session_states():
    """Declare all session state variables."""
    st.session_state.clip_ebc_model = ClipEBCOnnx(onnx_model_path="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/CLIP_EBC_nwpu_rmse_onnx.onnx")
    init_video_state()
    init_ebc_image_state()
    init_falldown_image_state()
    init_falldown_video_state()

def init_video_state():
    """Initialize Video session state variables."""
    
    st.session_state.video_path = None
    st.session_state.time_interval = None

    st.session_state.alarm_green_value_bar = 50
    st.session_state.alarm_orange_value_bar = 100
    st.session_state.alarm_red_value_bar = 150    

    st.session_state.alarm_green_value = 50
    st.session_state.alarm_orange_value = None
    st.session_state.alarm_red_value = None
    st.session_state.model_load_time = None
    st.session_state.fps_time_sleep = None
    st.session_state.count_ui = None

    st.session_state.progress_duration = 10 # 10초
    st.session_state.infer_q_size = None # 10초

    st.session_state.setdefault("origin", None)
    st.session_state.setdefault("dense", None)
    st.session_state.setdefault("dot", None)

    st.session_state.image_path_falldown = None
    st.session_state.video_path_falldown = None

    st.session_state.frame_queue = Queue(maxsize=1500)
    st.session_state.frame_queue_infer = Queue(maxsize=300)
    st.session_state.frame_queue_dense_result = Queue(maxsize=300)
    st.session_state.frame_queue_dot_result = Queue(maxsize=300)
    st.session_state.frame_queue_count_result = Queue(maxsize=300)


def init_ebc_image_state():
    """Initialize Video session state variables."""
    st.session_state.image_count_path = None

    st.session_state.image_dense_output = None
    st.session_state.image_dot_output = None
    st.session_state.image_count_output = None
    st.session_state.image_count_result = None
    st.session_state.original_image = None


def init_falldown_image_state():
    """Initialize Video session state variables."""
    st.session_state.image_falldown_path = None

    st.session_state.image_falldown_output = None
    st.session_state.image_falldown_description_output = None
    st.session_state.image_falldown_alarm_output = None


def init_falldown_video_state():
    """Initialize Video session state variables."""
    st.session_state.video_falldown_path = None

    st.session_state.video_falldown_output = None
    st.session_state.video_falldown_description_output = None
    st.session_state.video_falldown_alarm_output = None
    st.session_state.progress_duration_vqa = None

    st.session_state.frame_queue_vqa = Queue(maxsize=1500)
    st.session_state.frame_queue_infer_vqa = Queue(maxsize=300)

    st.session_state.frame_queue_infer_result_vqa = Queue(maxsize=300)
    st.session_state.frame_queue_desc_result_vqa = Queue(maxsize=300)
    st.session_state.frame_queue_alarm_result_vqa = Queue(maxsize=300)


