import urllib.parse
from fastapi import Depends, HTTPException, Request, status
import httpx
import requests
import sys
import argparse
from HandleUpload import  *

# --- 日志记录功能 ---
start_time = None
step_times = {}

# 飞书应用信息
APP_ID = "cli_a8e582b00539900b"
APP_SECRET = "iJ9jl1Oz0NnRiquHpsm9IedofHVzH5VQ"
REDIRECT_URI = "http://localhost:8000/call_back"


# --- API接口部分 ---
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form,Request
    from fastapi.responses import JSONResponse,RedirectResponse
    from pydantic import BaseModel
    from fastapi.middleware.cors import CORSMiddleware
    from utils import clean_text, fetch_webpage_content
    from feishu_api import get_feishu_doc_content
    import traceback
    import uvicorn
    import os
    import json
    import time
    import datetime
    from starlette.middleware.sessions import SessionMiddleware
    
    # 创建FastAPI应用
    app = FastAPI(
        title="测试用例比较工具API",
        description="比较AI生成的测试用例与黄金标准测试用例，评估测试用例质量",
        version="1.0.0"
    )
    
    # 允许跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有HTTP头
    )


    @app.get("/")
    async def root():
        """API根路径，返回基本信息"""
        return JSONResponse(content={
            "name": "测试用例比较工具API",
            "version": "1.0.0",
            "description": "比较AI生成的测试用例与黄金标准测试用例，评估测试用例质量"
        })


    def require_header_token(request: Request):
        # token = request.headers.get("Authorization")
        token = request.cookies.get("access_token")
        if not token:
            print("require_token: 没有找到access_token")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        print("require_token: 找到access_token:", token)
        return token


    @app.post("/test")
    async def test(request: Request, token: str = Depends(require_header_token)):
        print("test token:", token)
        return JSONResponse(status_code=200, content={"message": "Test successful", "token": token})

    # 上传接口
    @app.post("/upload_doc")
    async def upload_doc(file: UploadFile = File(...)):
        filename = file.filename.lower()
        if not (filename.endswith(".pdf") or filename.endswith(".docx")):
            raise HTTPException(status_code=400, detail="仅支持 .docx 和 .pdf 文件")

        file_bytes = await file.read()

        try:
            extracted_text = await extract_markdown(file_bytes, filename)
            print(extracted_text)
        except Exception as e:
            print(f"文档解析失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"文档解析失败: {str(e)}")

        try:
            llm_response = await call_deepseek_llm(extracted_text)
            return {
                "success": True,
                "json": llm_response["json"]
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"模型调用失败: {str(e)}")


    class TextRequest(BaseModel):
        text: str


    class ImageRequest(BaseModel):
        text: str  # 需求文本
        image_name: str
        image_base64: str  # Base64 编码的图片字符串

    @app.post("/upload_img")
    async def upload_img(request: ImageRequest):
        try:
            # 提取请求中的文本和图片数据
            text = request.text
            image_name = request.image_name
            image_base64 = request.image_base64

            # 将图片和文本传递给 call_doubao_llm 函数
            response = await call_doubao_llm(text, image_name, image_base64)

            # 返回生成的结果
            return {
                "success": True,
                "markdown": response["markdown"],
                "json": response["json"]
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


    @app.post("/generate_from_text")
    async def generate_from_text(data: TextRequest):
        if not data.text or len(data.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="输入文本不能为空或太短")

        try:
            result = await call_deepseek_llm(data.text)
            print(result)
            return {
                "success": True,
                "markdown": result["markdown"],
                "json": result["json"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型调用失败: {str(e)}")

    # token: str = Depends(require_header_token)
    # get_user_name
    @app.get("/get_user_name")
    async def get_user_name(request: Request, access_token: str = Depends(require_header_token)):
        # access_token = request.headers.get("Authorization")
        url = "https://open.feishu.cn/open-apis/authen/v1/user_info"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            user_resp = requests.get(url, headers=headers)
            user_data = user_resp.json()["data"]["open_id"]
            user_name = user_resp.json()["data"]["name"]
            print(user_data)
            print(user_data)
            return JSONResponse(
                status_code=200,
                content={
                    "open_id": user_data,
                    "user_name": user_name,
                    "access_token": access_token,
                }
            )
        except Exception as e:
            # 鉴权失败，重新登陆
            return JSONResponse(status_code=401, content={"error": "鉴权失败，请重新登录"})

    @app.get("/login")
    async def login():
        redirect_uri = 'http://localhost:8000/call_back'
        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe="")
        auth_url = f'https://open.feishu.cn/open-apis/authen/v1/index?app_id={APP_ID}&redirect_uri={encoded_redirect_uri}&scope=docx:document drive:drive'
        response = RedirectResponse(
            auth_url,
            status_code=302
        )
        return response

    @app.get("/login_out")
    async def login_out(request: Request):
        response = RedirectResponse(
            "http://localhost:5173/LLMGenerate",
            status_code=302
        )
        response.delete_cookie("access_token")
        return response

    @app.get("/call_back")
    async def auth_callback(request: Request):
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="缺少 code")

        token_url = "https://open.feishu.cn/open-apis/authen/v2/oauth/token"
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": APP_ID,
            "client_secret": APP_SECRET,
            "redirect_uri": REDIRECT_URI
        }

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(token_url, json=payload)
            token_data = token_resp.json()

            if token_data.get("code", 0) != 0:
                return JSONResponse(status_code=500, content={"error": token_data.get("msg", "token 获取失败")})
            access_token = token_data["access_token"]


            redirect_url = request.query_params.get("state") or "http://localhost:5173"

            response = RedirectResponse(redirect_url)
            # 2小时
            response.set_cookie("access_token", access_token, httponly=True, secure=True, expires=2 * 60 * 60)
            return response


except ImportError as e:
    # 检测未安装的包
    import traceback
    traceback.print_exc()
    app = None

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # 命令行模式
        parser = argparse.ArgumentParser(description="测试用例比较工具")
        parser.add_argument("--ai", help="AI生成的测试用例文件路径")
        parser.add_argument("--golden", help="黄金标准测试用例文件路径")
        args = parser.parse_args(sys.argv[2:])
    else:
        # API模式（默认）
        if app:
            import uvicorn
            uvicorn.run("GenerateAndCompareCasesAPI:app", host="127.0.0.1", port=8000, reload=True)
        else:
            print("运行错误")