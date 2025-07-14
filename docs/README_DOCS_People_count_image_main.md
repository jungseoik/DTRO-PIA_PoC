
## 인터페이스 소개 (Crowd People Counting)

본 시스템은 이미지 또는 영상을 업로드하면 군중 수를 추론하여 시각적으로 결과를 제공합니다. 아래는 **People Count Image** 탭에서 이미지 파일을 업로드했을 때의 예시 화면입니다.

### 📌 업로드 및 추론 결과 예시

![Crowd Counting UI 1](docs/images/people_cnt_main.png)

- 좌측: 이미지 업로드 영역 (`JPG`, `PNG`, `BMP`, `WEBP` 형식 지원, 최대 200MB)
- 우측: 탐지된 인원 수와 위험도 상태 표시  
  → 예시: `Detected People: 221`, **위험 상태**로 표시됨

---

### 🔍 결과 시각화 (Original / Dense MAP / Dot MAP)

업로드 후에는 다음과 같은 3가지 형태로 시각화된 결과가 출력됩니다.

![Crowd Counting UI 2](docs/images/people_cnt_main2.png)

- **Original**: 업로드한 원본 이미지  
- **Dense MAP**: 밀도 기반 히트맵 시각화  
- **Dot MAP**: 각 인원 탐지 지점을 점으로 표시

