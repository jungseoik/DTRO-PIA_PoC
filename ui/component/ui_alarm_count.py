import streamlit as st

def render_count_status_ui(count, frame_idx, container):
    status = ""
    color = ""
    emoji = ""
    count = int(count)
    if count >= st.session_state.alarm_red_value:
        status = "🚨 **위험**"
        color = "red"
        emoji = "🔴"
    elif count >= st.session_state.alarm_orange_value:
        status = "⚠️ **경고**"
        color = "orange"
        emoji = "🟠"
    else:
        status = "✅ **안전**"
        color = "green"
        emoji = "🟢"

    container.markdown(f"""
    <div style="padding: 1rem; border-radius: 1rem; background-color: {color}; color: white; text-align: center;">
        <h3 style="margin-bottom: 0.5rem;">{emoji} Detected People: {count}</h3>
        <p style="margin: 0;">Frame Index: {frame_idx} | Status: {status}</p>
    </div>
    """, unsafe_allow_html=True)


def render_image_count_status_ui(count, container):
    status = ""
    color = ""
    emoji = ""
    count = int(count)
    if count >= st.session_state.alarm_red_value:
        status = "🚨 **위험**"
        color = "red"
        emoji = "🔴"
    elif count >= st.session_state.alarm_orange_value:
        status = "⚠️ **경고**"
        color = "orange"
        emoji = "🟠"
    else:
        status = "✅ **안전**"
        color = "green"
        emoji = "🟢"

    container.markdown(f"""
    <div style="padding: 1rem; border-radius: 1rem; background-color: {color}; color: white; text-align: center;">
        <h3 style="margin-bottom: 0.5rem;">{emoji} Detected People: {count}</h3>
        <p style="margin: 0;"> | Status: {status}</p>
    </div>
    """, unsafe_allow_html=True)