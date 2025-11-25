import requests
import json

url = "http://127.0.0.1:8000/v1/chat/completions"

data = {
    "messages": [
        {"role": "user", "content": "你好，请详细介绍一下你自己，不要思考，直接回答。"}
    ],
    "max_tokens": 32768,    # 之前断掉是因为这里设置太小了
    "temperature": 0.7,
    "stream": True,        # 开启流式
    "enable_thinking": False # 显式请求关闭思考
}

print("📡 正在连接流式 API...")
print("-" * 50)

# 关键：设置 stream=True
response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    # 按行读取服务器发送的数据
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            # SSE 格式通常以 "data: " 开头
            if line.startswith("data: "):
                json_str = line[6:] # 去掉 "data: "
                if json_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    # 获取增量内容
                    delta = chunk['choices'][0]['delta'].get('content', '')
                    if delta:
                        # flush=True 确保立即打印到屏幕，不缓存
                        print(delta, end="", flush=True)
                except json.JSONDecodeError:
                    pass
    print("\n" + "-" * 50)
    print("✅ 回答结束")
else:
    print("❌ 请求失败:", response.text)