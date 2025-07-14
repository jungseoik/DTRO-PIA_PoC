import streamlit as st


def siderbar_setting_ui():
    st.session_state.time_interval = st.sidebar.slider(
    "Inference Interval (frames)", min_value=1, value=30, step=1
        )   
    orange = st.sidebar.slider("🟠 경고 기준 (orange max)",
                    min_value=1, max_value=500,
                    value=st.session_state.alarm_orange_value_bar, step=1)
    red = st.sidebar.slider("🔴 위험 기준 (red max)", min_value=orange+1, max_value=1000,
                    value=st.session_state.alarm_red_value_bar, step=1)
    
    st.session_state.alarm_orange_value = orange
    st.session_state.alarm_red_value = red
    st.session_state.progress_duration = st.sidebar.slider(
        "Progress Bar Duration (seconds)",
        min_value=1, max_value=60,
        value=10,
        step=1)