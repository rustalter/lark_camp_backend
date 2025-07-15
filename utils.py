import re
import httpx
from bs4 import BeautifulSoup


# 清洗文本内容
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# 网页图文内容提取
async def fetch_webpage_content(url: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")

    # 提取文本内容
    tags = soup.find_all(['p', 'div', 'li', 'span'])
    text = "\n".join(tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True))

    # 提取图片链接
    images = soup.find_all("img")
    image_urls = [img.get("src") for img in images if img.get("src")]

    return {
        "text": clean_text(text),
        "images": image_urls[:100]
    }
