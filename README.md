# DTRO & PIA-PoC

<p align="center">
  <span style="display: inline-flex; align-items: center;">
    <img alt="PIA Logo" src="https://raw.githubusercontent.com/jungseoik/DTRO-PIA_PoC/main/assets/pia-logo-dark.png" height="60" style="margin-right: 8px;">
    <span style="font-size: 80px; color: #555; margin-right: 8px;">&</span>
    <img alt="DTRO Logo" src="https://raw.githubusercontent.com/jungseoik/DTRO-PIA_PoC/main/assets/dtro.svg" height="60">
  </span>
</p>

---

<p align="center">
  <img alt="Main Page" src="https://raw.githubusercontent.com/jungseoik/DTRO-PIA_PoC/main/assets/main_page.png" style="max-width: 100%;">
</p>

---

## PIA-SPACE와 DTRO가 협업한 PoC 프로젝트의 공식 구현 레포지토리입니다.

본 저장소는 사람 수 카운팅, 실시간 에스컬레이터 쓰러짐 탐지 기능의 프론트엔드 및 백엔드 구성 예제를 포함하며,  
실제 프로토타입 시연에 활용된 코드를 정리한 것입니다.

## Install

```bash
conda create -n dtro python==3.11 -y
conda activate dtro
pip install -r requirements.txt
```

### 실행순서:
- [TRTLLM Server API Documentation](docs/README_TRTLLM_SERVER.md)
- [HF ONNX Down Documentation](docs/README_onnx_download.md)

```bash
위에 전부 완료되면

streamlit run homepage.py

```

### UI 사용 설명:
- [UI - People Count Image Documentation](docs/README_DOCS_People_count_image_main.md)
- [UI - People Count Video Documentation](docs/README_DOCS_People_count_video_main.md)
- [UI - VQA Image Documentation](docs/README_DOCS_vqa_image_main.md)
- [UI - VQA Video Documentation](docs/README_DOCS_vqa_video_main.md)