import streamlit as st
from PIL import Image
import base64 

def get_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def logo_header_ui():

    dtro_logo_svg = get_image_as_base64("/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/dtro.svg")
    pia_log_png = get_image_as_base64("/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/pia-logo-white.png")

    # logo_dark = Image.open('assets/pia-logo-dark.png') 
    # logo_white = Image.open('assets/pia-logo-white.png') 
    
    st.markdown(
        f"""
    <h1 style="margin: 0;">PIA-SPACE & DTRO</h1>
    <div style="display:flex;
                align-items:center;         
                justify-content:flex-end;    
                width:100%;">              
        {f'<img src="data:image/svg+xml;base64,{dtro_logo_svg}" height="50" style="margin-right:15px;">' if dtro_logo_svg else ""}
        {f'<img src="data:image/png;base64,{pia_log_png}" height="50">' if pia_log_png else ""}
    </div>
        """,
        unsafe_allow_html=True,
    )
