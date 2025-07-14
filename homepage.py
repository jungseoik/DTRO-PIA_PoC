import streamlit as st
from ui.tabs.tab_count_video import count_video_tab
from ui.tabs.tab_count_image import count_image_tab
from ui.tabs.tab_falldown_image import vqa_image_tab
from ui.tabs.tab_falldown_video import vqa_video_tab

from PIL import Image
from ui.init.declare_manager import declare_all_session_states
from ui.component.ui_logo_header import logo_header_ui

declare_all_session_states()
logo_dark = Image.open('assets/pia-logo-dark.png') 
logo_white = Image.open('assets/pia-logo-white.png') 

st.set_page_config(
    page_title="DTRO",
    page_icon=logo_dark,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.sidebar.image(logo_white)
logo_header_ui()
tab_count_video, tab_count_image, tab_falldown_image, tab_falldown_video = st.tabs(["People Count Video", "People Count Image" , "Falldown image", "falldown video"])

with tab_count_video:
    count_video_tab()
with tab_count_image:
    count_image_tab()
with tab_falldown_image:
    vqa_image_tab()
with tab_falldown_video:
    vqa_video_tab()


