import asyncio
import json
import time
from threading import Thread

import torch
import gc
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modelscope import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextIteratorStreamer # 引入迭代流式工具

# ================== 1. 模型加载 (保持不变) ==================
app = FastAPI(title="Qwen3-14B Stream API", description="Local LLM Server")

model_name = "Qwen/Qwen3-14B" # 保持你的路径

print(f"🚀 正在启动服务端，加载模型: {model_name} ...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 清理显存
gc.collect()
torch.cuda.empty_cache()
print(f"✅ 模型加载完毕！显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ================== 2. 定义请求结构 ==================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 32768    # 默认设大一点，防止回答一半断掉
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True       # 默认开启流式
    enable_thinking: bool = False # 默认关闭思考模式

# ================== 3. 核心逻辑：流式生成器 ==================
def stream_generation(inputs, streamer, max_tokens, temp, top_p):
    """在独立线程中运行推理，把结果喂给 streamer"""
    try:
        model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            streamer=streamer,
            temperature=temp,
            top_p=top_p,
            do_sample=True
        )
    except Exception as e:
        print(f"生成出错: {e}")

async def generate_stream_response(streamer):
    """异步读取 streamer 中的 token 并按 SSE 格式发送"""
    request_id = f"chatcmpl-{int(time.time())}"
    
    for new_text in streamer:
        if not new_text: continue
        
        # 构建 OpenAI 兼容的流式数据包
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": new_text},
                "finish_reason": None
            }]
        }
        # SSE 格式要求：以 data: 开头，双换行结尾
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0) # 让出控制权

    # 发送结束信号
    end_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# ================== 4. API 接口 ==================
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    print(f"📩 收到请求 (Stream={request.stream}, Thinking={request.enable_thinking})")
    
    # 1. 转换消息并应用模板
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # --- 关键修改：在此处关闭思考模式 ---
    # enable_thinking=False 传给模板，防止生成 <think> 标签
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=request.enable_thinking 
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 2. 初始化流式迭代器
    # skip_prompt=True: 不重复打印问题
    # skip_special_tokens=True: 不打印 <|endoftext|> 等特殊符
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 3. 启动子线程进行推理 (这是流式的关键！)
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_tokens=request.max_tokens,
        temp=request.temperature,
        top_p=request.top_p
    )
    
    thread = Thread(target=stream_generation, kwargs=generation_kwargs)
    thread.start()

    # 4. 如果是流式请求，返回 StreamingResponse
    if request.stream:
        return StreamingResponse(
            generate_stream_response(streamer), 
            media_type="text/event-stream"
        )
    
    # 5. 如果非流式 (兼容旧代码)，等待线程结束收集所有文本
    else:
        full_response = ""
        for new_text in streamer:
            full_response += new_text
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop"
            }]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)