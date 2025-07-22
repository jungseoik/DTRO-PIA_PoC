
## PIA AI Package - 모델 다운로드 필수 가이드

- 본 프로젝트는 실행 시 Hugging Face Hub에서 모델을 다운로드합니다.
- 따라서 반드시 토큰 인증을 진행해주셔야 합니다.
---

## Hugging Face 토큰 인증

- 모델이 **private** repository에 있을 경우, Hugging Face 액세스 토큰 인증이 필요합니다.

### 1. Hugging Face 계정 생성 및 로그인

- [https://huggingface.co/join](https://huggingface.co/join) 에서 계정을 생성하거나 로그인.

### 2. 액세스 토큰 생성

- 프로필 → Settings → [Access Tokens](https://huggingface.co/settings/tokens) 에서 **"New token"** 생성
- 권한은 **"Read"** 권한으로 충분.

### 3. CLI를 통해 토큰 로그인

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

- 위 명령 실행 후, 터미널에 토큰을 입력.
- 한 번 로그인하면 캐시에 저장되어 재인증 없이 다운로드.

>💡 CLI 로그인은 .huggingface/token에 저장되므로 재부팅 후에도 유지됩니다.
