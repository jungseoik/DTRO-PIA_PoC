import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

def temp_file_save(
    file_upload_obj: st.runtime.uploaded_file_manager.UploadedFile,
    overwrite=False
) -> str:
    """
    업로드된 이미지 파일을 임시 디렉토리에 저장하고 파일 경로를 반환하는 함수.
    동일한 파일을 다시 업로드 할 경우 체크 한 뒤 해당 파일 활용

    Args:
        file_upload_obj (st.runtime.uploaded_file_manager.UploadedFile): Streamlit에서 업로드된 파일 객체.

    Returns:
        str: 임시 디렉토리에 저장된 파일의 경로.
    """
    destination_path = Path(tempfile.gettempdir()) / file_upload_obj.name
    if not destination_path.exists() or overwrite:
        with open(destination_path, "wb") as f:
            f.write(file_upload_obj.getvalue())
            print(f"image file has been saved to {destination_path}")
    return str(destination_path)


def local_image_input() -> None:
    """
    Streamlit 파일 업로드 컴포넌트를 사용하여 이미지 파일을 업로드 받고,
    세션 상태에 파일 경로를 저장하는 함수.

    Returns:
        None
    """
    image_file_path = st.file_uploader(
        "이미지 파일 업로드",
        key="st_image_file_path",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if image_file_path:
        image_file = temp_file_save(image_file_path)
        st.session_state.image_count_path = image_file
    else:
        image_file = None
        st.session_state.image_count_path = None


def local_image_vaq_input() -> None:
    """
    Streamlit 파일 업로드 컴포넌트를 사용하여 이미지 파일을 업로드 받고,
    세션 상태에 파일 경로를 저장하는 함수.

    Returns:
        None
    """
    image_file_path = st.file_uploader(
        "이미지 파일 업로드",
        key="st__vaq_image_file_path",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if image_file_path:
        image_file = temp_file_save(image_file_path)
        st.session_state.image_falldown_path = image_file
    else:
        image_file = None
        st.session_state.image_falldown_path = None
