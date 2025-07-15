import json

import requests

# resp = requests.post('https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/',
#   json={"app_id": "cli_a8ef8e2bca7bd01c", "app_secret": "7WJD5NwtsIDfGhwRhI6HEfmlWAULqQA5"})
# tenant_token = resp.json()['tenant_access_token']
# print(tenant_token)
# resp = requests.get('https://open.feishu.cn/open-apis/drive/root/v2/my/root_folder_space',
#   headers={"Authorization": f"Bearer {tenant_token}"})
# # root_folder_token = resp.json()['data']['root_folder_token']
# resp = requests.get(
#   f'https://open.feishu.cn/open-apis/drive/explorer/v2/file/{document_id}/meta',
#   headers={"Authorization": f"Bearer {tenant_token}"}
# )
# doc_token = resp.json()['data']['token']
# print(resp.json())
import urllib.parse
APP_ID = "cli_a8ef8e2bca7bd01c"
APP_SECRET = "7WJD5NwtsIDfGhwRhI6HEfmlWAULqQA5"
redirect_uri = 'http://localhost:8000/call-back'
encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe="")
auth_url = f'https://open.feishu.cn/open-apis/authen/v1/index?app_id={APP_ID}&redirect_uri={encoded_redirect_uri}'
print("https://open.feishu.cn/open-apis/authen/v1/index?app_id=cli_a8ef8e2bca7bd01c&redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fcall-back" == auth_url)
print(auth_url)

print("https://open.feishu.cn/open-apis/authen/v1/index?app_id=cli_a8ef8e2bca7bd01c&redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fcall-back" == auth_url)

print("https://open.feishu.cn/open-apis/authen/v1/index?app_id=cli_a8ef8e2bca7bd01c&redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fcall-back" == auth_url)

# response = requests.get("http://127.0.0.1:8000/login", allow_redirects=False)
# print(response.headers["Location"])
# response = requests.get(response.headers["Location"], allow_redirects=False)
# print(response.headers["Location"].replace("accounts", "open"))
s = set()
s.add(1)
print(json.dumps(s))