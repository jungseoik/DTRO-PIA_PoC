
## **Frame Scoring API (base64 JSON) 문서**

이 API는 동영상 프레임 이미지들을 받아 각 프레임의 점수를 계산(추론)하고 결과를 반환합니다. 모든 이미지 데이터는 **JSON 본문 내에서 Base64로 인코딩된 문자열** 형태로 전달받습니다.

### **엔드포인트 정보**

  * **URL:** `/predict_json`
  * **Method:** `POST`
  * **Content-Type:** `application/json`

-----

###  **요청 (Request)**

클라이언트는 아래의 구조를 가진 JSON 객체를 `POST` 요청의 본문(body)에 담아 전송해야 합니다.

#### **요청 본문 구조**

```json
{
  "video_name": "string",
  "frames": [
    {
      "frame_index": "integer (>= 0)",
      "data": "string (base64-encoded image)"
    }
  ]
}
```

#### **필드 설명**

| 필드명 | 타입 | 필수 여부 | 설명 |
| :--- | :--- | :--- | :--- |
| `video_name` | string | 예 | 프레임들이 속한 동영상의 이름 또는 식별자입니다. |
| `frames` | list of objects | 예 | 점수를 계산할 프레임 정보가 담긴 리스트입니다. **비어 있으면 안 됩니다.** |
| ⇨ `frame_index` | integer | 예 | 해당 프레임의 순서 번호(인덱스)입니다. `0` 이상의 정수여야 합니다. |
| ⇨ `data` | string | 예 | 프레임 이미지를 **Base64로 인코딩한 문자열**입니다. 유효한 Base64 형식이 아니면 오류가 발생합니다. |

#### **요청 예시 (Request Example)**

```json
{
  "video_name": "test_video_01.mp4",
  "frames": [
    {
      "frame_index": 0,
      "data": "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A/v4oAKKKKAP/2Q=="
    },
    {
      "frame_index": 15,
      "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    }
  ]
}
```

-----

###  **응답 (Response)**

요청이 성공적으로 처리되면, API는 `200 OK` 상태 코드와 함께 아래 구조의 JSON 객체를 반환합니다.

#### **응답 본문 구조**

```json
{
  "video_name": "string",
  "results": [
    {
      "frame_index": "integer",
      "result": "float"
    }
  ]
}
```

#### **필드 설명**

| 필드명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `video_name` | string | 요청 시 보냈던 동영상 이름이 그대로 반환됩니다. |
| `results` | list of objects | 각 프레임의 추론 결과가 담긴 리스트입니다. |
| ⇨ `frame_index` | integer | 요청했던 프레임의 인덱스입니다. |
| ⇨ `result` | float | 해당 프레임에 대해 모델이 추론한 점수(score)입니다. |

#### **응답 예시 (Response Example)**

```json
{
  "video_name": "test_video_01.mp4",
  "results": [
    {
      "frame_index": 0,
      "result": 0.8734
    },
    {
      "frame_index": 15,
      "result": 0.1298
    }
  ]
}
```

-----

###  **오류 응답 (Error Responses)**

요청이 잘못되었거나 서버 처리 중 문제가 발생하면 `200`이 아닌 상태 코드와 함께 오류 정보가 담긴 JSON이 반환됩니다.

  * **`400 Bad Request`**:
      * `frame_index`의 이미지를 디코딩할 수 없을 때 발생합니다. (예: Base64 문자열 손상)
      * `detail` 필드에 어떤 프레임에서 문제가 발생했는지 표시됩니다.
  * **`422 Unprocessable Entity`**:
      * 요청 본문이 Pydantic 스키마의 유효성 검사를 통과하지 못했을 때 발생합니다.
      * `frames` 리스트가 비어있거나, 필드 타입이 맞지 않거나, 필수 필드가 누락된 경우입니다.
  * **`500 Internal Server Error`**:
      * 모델 추론 과정 등 서버 내부 로직에서 예측하지 못한 오류가 발생했을 때 반환됩니다.