
# InternVL3-2B Tensorrt llm 서버 실행 가이드

> **전제조건**: 이미 모든 모델 빌드가 완료된 상태에서 서버만 실행하는 가이드입니다.

## 0. 환경 준비

```bash
git clone https://github.com/NetEase-Media/grps_trtllm.git
cd grps_trtllm
git submodule update --init --recursive
```

## 1. 컨테이너 실행

```bash
# 컨테이너 생성 및 실행
docker run -itd --name ivl3_2b_server --runtime=nvidia --network host \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/grps_dev -v /tmp:/tmp -w /grps_dev \
  registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12 bash

# 컨테이너 접속
docker exec -it ivl3_2b_server bash
```

## 2. 모델 다운로드 & 의존 패키지
```bash
# 2B 클론
apt update && apt install -y git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3-2B /tmp/InternVL3-2B

## A100 진행시 그대로 진행 
#  ./tools/internvl2/requirements.txt 위치 종속성중
#   pycuda==2025.1.1  로 수정
pip install -r ./tools/internvl2/requirements.txt   # ViT 변환 스크립트용

```
## 3. 체크포인트 변환
```bash
rm -rf /tmp/InternVL3-2B/tllm_checkpoint/
python3 tools/internvl2/convert_qwen2_ckpt.py \
        --model_dir  /tmp/InternVL3-2B \
        --output_dir /tmp/InternVL3-2B/tllm_checkpoint/ \
        --dtype bfloat16 --load_model_on_cpu

```
## 4. **LLM 엔진** 빌드 (batch size 1)

```bash
rm -rf /tmp/InternVL3-2B/trt_engines/
trtllm-build \
  --checkpoint_dir /tmp/InternVL3-2B/tllm_checkpoint/ \
  --output_dir     /tmp/InternVL3-2B/trt_engines/ \
  --gemm_plugin bfloat16 \
  --max_batch_size 1 \
  --paged_kv_cache enable \
  --use_paged_context_fmha enable \
  --max_input_len 30720 \
  --max_seq_len   32768 \
  --max_num_tokens 32768 \
  --max_multimodal_len 26624        

  모델 빌드 용량이 달라짐 대략 어느정도 토큰 소모되는지 파악하고 빌드하기 바랍니다.
  위에 세팅은 20GB정도 잡아먹는 빌드
  아래 내용 실행 추천
--------------------------------------
rm -rf /tmp/InternVL3-2B/trt_engines/
trtllm-build \
  --checkpoint_dir /tmp/InternVL3-2B/tllm_checkpoint/ \
  --output_dir /tmp/InternVL3-2B/trt_engines/ \
  --gemm_plugin bfloat16 \
  --max_batch_size 1 \
  --paged_kv_cache enable \
  --use_paged_context_fmha enable \
  --max_input_len 4096 \
  --max_seq_len 4096 \
  --max_num_tokens 4608 \
  --max_multimodal_len 4608

```
---

## 6. **ViT 엔진** 빌드

- `-maxBS` 는 *이미지 패치* 동시 처리 기준이라 그대로 26 유지.

```bash
python3 tools/internvl2/build_vit_engine.py \
  --pretrainedModelPath /tmp/InternVL3-2B \
  --imagePath           ./data/frames/frame_0.jpg \
  --onnxFile            /tmp/InternVL3-2B/vision_encoder_bfp16.onnx \
  --trtFile             /tmp/InternVL3-2B/vision_encoder_bfp16.trt \
  --dtype bfloat16 \
  --minBS 1 --optBS 13 --maxBS 26
```
--------------------------------------------------------------

## 7. `inference.yml` 수정

`conf/inference_internvl3.yml` 기준으로 **경로와 batch size**

```yaml
inferer_args:
  llm_style: internvl3          # 그대로
  tokenizer_path: /tmp/InternVL3-2B            # ★
  gpt_model_path: /tmp/InternVL3-2B/trt_engines # ★
  gpt_model_type: inflight_fused_batching
  ...
  # 배치 1이므로 스케줄러 정책이나 토큰 제한을 조정할 필요는 없음

```

## 8. 패키징 & 서버 기동

### 기존 아카이브 파일이 있는 경우
```bash
# 아카이브 파일 확인
ls -la server.mar

# 바로 서버 시작
grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml

# 서버 상태 확인
grpst ps
```

### 아카이브 파일이 없는 경우
```bash
# 새로운 아카이브 생성
grpst archive .

# 서버 시작
grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml

# 서버 상태 확인
grpst ps
```

## 9. 서버 테스트

서버가 정상적으로 실행되었는지 테스트:

```bash
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "<image>\nexplain this image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-2B/examples/image1.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'

  
```

## 10. 서버 관리 명령어

```bash
# 서버 상태 확인
grpst ps

# 서버 중지
grpst stop

# 서버 재시작
grpst restart

# 로그 확인
grpst logs
```