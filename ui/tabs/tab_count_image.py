
import streamlit as st
from ui.component.ui_image_input import local_image_input
from ui.component.ui_alarm_count import render_image_count_status_ui

def count_image_tab():

    col_ebc_image_input, col_ebc_count_output = st.columns(2, gap="medium")
    with col_ebc_image_input:
        st.subheader("Image Input")
        local_image_input()
    with col_ebc_count_output:
        st.subheader("Peoplc Count")
        st.session_state.image_count_output = st.empty()

    button_ebc_image =  st.button("Run Inference")
    st.divider()

    col_original_image, col_dense_image, col_dot_image= st.columns(3, gap="medium")
    with col_original_image:
        st.subheader("Original")
        st.session_state.original_image = st.empty()
    with col_dense_image:
        st.subheader("Dense MAP")
        st.session_state.image_dense_output = st.empty()
    with col_dot_image:
        st.subheader("Dot MAP")
        st.session_state.image_dot_output = st.empty()

    if button_ebc_image:
        if not st.session_state.image_count_path:
            st.warning("먼저 이미지를 업로드 해주세요.")
        else:
            model = st.session_state.clip_ebc_model
            st.session_state.image_count_result = model.predict(st.session_state.image_count_path)
            render_image_count_status_ui(st.session_state.image_count_result, st.session_state.image_count_output)

            _ , dense_map = model.visualize_density_map()
            _ , dot_map = model.visualize_dots()
            st.session_state.original_image.image(st.session_state.image_count_path)
            st.session_state.image_dense_output.image(dense_map)
            st.session_state.image_dot_output.image(dot_map)
            

