import os
import httpx
from typing import List, Optional

ARK_API_KEY = os.environ.get("ARK_API_KEY") or "f3fbd54b-1775-4250-be19-528cf14f1291"
ARK_MODEL_ID = "deepseek-r1-250120"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


async def call_model(prompt: str, img_urls: Optional[List[str]] = None) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARK_API_KEY}"
    }

    message_content = []

    if img_urls:
        for url in img_urls:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

    message_content.append({
        "type": "text",
        "text": prompt
    })

    payload = {
        "model": ARK_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": message_content
            }
        ]
    }

    try:
        async with httpx.AsyncClient(timeout=1000.0) as client:
            response = await client.post(f"{ARK_BASE_URL}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
    except httpx.HTTPStatusError as e:
        # HTTP响应错误
        print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        # 请求连接错误
        print(f"Request error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
