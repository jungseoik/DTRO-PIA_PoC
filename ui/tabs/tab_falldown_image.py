

import streamlit as st
from ui.component.ui_image_input import local_image_vaq_input
from utils.api.vqa_api import internvl_vision_api_response
from env.config import PROMPT_V3 , PROMPT_V2

def render_description_section():
    """스타일리시한 Description 섹션 렌더링"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h3 style="color: white; margin: 0; font-weight: 600;">📝 Description</h3>
    </div>
    """, unsafe_allow_html=True)
    return st.empty()

def render_alarm_section():
    """스타일리시한 Alarm 섹션 렌더링"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 10px;">
        <h3 style="color: white; margin: 0; font-weight: 600;">🚨 Alarm Status</h3>
    </div>
    """, unsafe_allow_html=True)
    return st.empty()

def display_description_result(container, desc_result):
    """Description 결과를 스타일리시하게 출력"""
    container.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                padding: 25px; border-radius: 15px; 
                border: 2px solid #667eea; 
                box-shadow: 0 8px 16px rgba(102, 126, 234, 0.15);
                margin-top: -10px; margin-bottom: 20px;">
        <p style="font-size: 18px; font-weight: 600; line-height: 1.7; 
                  color: #2c3e50; margin: 0; text-align: justify;
                  text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
            {desc_result}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_alarm_result(container, category_result):
    """Alarm 결과를 상태에 따라 스타일리시하게 출력"""
    if category_result.lower() == "normal":
        alarm_style = """
        <div style="background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);">
            <span style="color: white; font-size: 18px; font-weight: 600;">
                ✅ {status}
            </span>
        </div>
        """
    else:
        alarm_style = """
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 15px; border-radius: 10px; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                    animation: pulse 2s infinite;">
            <span style="color: white; font-size: 18px; font-weight: 600;">
                ⚠️ {status}
            </span>
        </div>
        <style>
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
        </style>
        """
    
    container.markdown(alarm_style.format(status=category_result), unsafe_allow_html=True)

def vqa_image_tab():
    col_vqa_image_input,  _ = st.columns(2, gap="medium")
    with col_vqa_image_input:
        st.subheader("VQA Input")
        local_image_vaq_input()

    button_vqa_image =  st.button("Inference")
    st.divider()

    col_vqa_origin_image, col_description_alarm = st.columns(2, gap="medium")
    with col_vqa_origin_image:
        st.subheader("Original")
        st.session_state.image_falldown_output = st.empty()
    with col_description_alarm:
        st.subheader("Output")
        st.session_state.image_falldown_description_output = render_description_section()
        st.session_state.image_falldown_alarm_output = render_alarm_section()

    if button_vqa_image:
        if not st.session_state.image_falldown_path:
            st.warning("먼저 이미지를 업로드 해주세요.")
        else:
           category_result, desc_result = internvl_vision_api_response(image_path=st.session_state.image_falldown_path, question = PROMPT_V2)
           
           # 함수를 사용해 스타일리시하게 결과 출력
           display_description_result(st.session_state.image_falldown_description_output, desc_result)
           display_alarm_result(st.session_state.image_falldown_alarm_output, category_result)
           st.session_state.image_falldown_output.image(st.session_state.image_falldown_path)
