import httpx

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


async def fetch_all_blocks(document_id: str, user_access_token: str):
    headers = {"Authorization": f"Bearer {user_access_token}"}
    url = f"{FEISHU_API_BASE}/docx/v1/documents/{document_id}/blocks"
    limit = 100
    cursor = None
    all_blocks = []

    async with httpx.AsyncClient() as client:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            items = data.get("data", {}).get("items", [])
            all_blocks.extend(items)

            if not data.get("data", {}).get("has_more", False):
                break
            cursor = data.get("data", {}).get("cursor")

    return all_blocks


async def get_single_image_url(file_token: str, user_access_token: str) -> str:
    headers = {"Authorization": f"Bearer {user_access_token}"}
    url = f"{FEISHU_API_BASE}/drive/v1/medias/batch_get_tmp_download_url"
    params = {"file_tokens": file_token}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
    for item in data.get("data", {}).get("tmp_download_urls", []):
        if item.get("file_token") == file_token:
            return item.get("tmp_download_url", "")
    return ""


def extract_text(elements):
    texts = []
    for elem in elements:
        if "text_run" in elem:
            texts.append(elem["text_run"]["content"])
        elif "equation" in elem:
            texts.append(elem["equation"]["expression"])
    return "".join(texts)


def parse_block(block, blocks_map, image_url_map):
    block_type = block.get("block_type")
    md = ""

    if block_type == 1:  # 页面 Block，递归子块
        for child_id in block.get("children", []):
            child = blocks_map.get(child_id)
            if child:
                md += parse_block(child, blocks_map, image_url_map)

    elif 3 <= block_type <= 9:  # heading2~heading8
        level = block_type - 2
        heading_key = f"heading{level}"
        if heading_key in block:
            text = extract_text(block[heading_key]["elements"])
            md += f"{'#' * level} {text}\n\n"
        for child_id in block.get("children", []):
            child = blocks_map.get(child_id)
            if child:
                md += parse_block(child, blocks_map, image_url_map)

    elif block_type == 2:  # 文本 Block
        if "text" in block:
            text = extract_text(block["text"]["elements"])
            md += f"{text}\n\n"

    elif block_type == 10:  # 无序列表 bullet
        if "bullet" in block:
            text = extract_text(block["bullet"]["elements"])
            md += f"- {text}\n"
        for child_id in block.get("children", []):
            child = blocks_map.get(child_id)
            if child:
                child_md = parse_block(child, blocks_map, image_url_map)
                child_md = "\n".join("  " + line if line.strip() else line for line in child_md.splitlines())
                md += child_md + "\n"

    elif block_type == 11:  # 有序列表 ordered
        if "ordered" in block:
            text = extract_text(block["ordered"]["elements"])
            md += f"1. {text}\n"
        for child_id in block.get("children", []):
            child = blocks_map.get(child_id)
            if child:
                child_md = parse_block(child, blocks_map, image_url_map)
                child_md = "\n".join("  " + line if line.strip() else line for line in child_md.splitlines())
                md += child_md + "\n"

    elif block_type == 27:  # 图片块
        image = block.get("image", {})
        file_token = image.get("file_token") or image.get("token")
        if not file_token and "origin" in image:
            file_token = image["origin"].get("file_token") or image["origin"].get("token")
        url = image_url_map.get(file_token, "")
        md += f"![image]({url})\n\n"

    elif block_type == 14:  # 代码块
        if "code" in block:
            text = extract_text(block["code"]["elements"])
            md += f"```\n{text}\n```\n\n"

    elif block_type == 15:  # 引用块
        if "quote" in block:
            text = extract_text(block["quote"]["elements"])
            md += f"> {text}\n\n"

    return md


def blocks_to_markdown(blocks, image_url_map):
    blocks_map = {b["block_id"]: b for b in blocks}
    roots = [b for b in blocks if not b.get("parent_id")]

    md_all = ""
    for root in roots:
        md_all += parse_block(root, blocks_map, image_url_map)

    return md_all


async def get_feishu_doc_content(document_id: str, user_access_token: str):
    blocks = await fetch_all_blocks(document_id, user_access_token)

    image_tokens = []
    for b in blocks:
        if b.get("block_type") == 27 and "image" in b:
            image = b["image"]
            file_token = image.get("file_token") or image.get("token")
            if not file_token and "origin" in image:
                file_token = image["origin"].get("file_token") or image["origin"].get("token")
            if file_token:
                image_tokens.append(file_token)

    image_url_map = {}
    for token in image_tokens:
        try:
            url = await get_single_image_url(token, user_access_token)
            image_url_map[token] = url
        except Exception as e:
            print(f"获取图片 {token} 下载链接失败: {e}")

    markdown = blocks_to_markdown(blocks, image_url_map)

    import re
    text_only = re.sub(r"!\[image\]\([^)]+\)", "", markdown).strip()

    return {
        "text": text_only,
        "markdown": markdown,
        "images": list(image_url_map.values())
    }
