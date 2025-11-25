from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextStreamer # <--- 引入流式输出工具
import torch
import gc

model_name = "Qwen/Qwen3-14B"

print(f"🚀 正在加载 Tokenizer: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("⚙️  正在配置 8-bit 量化模式...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("📥 正在加载模型 (强制使用 GPU)...")
# 强制指定 device_map="cuda:0"，确保 100% 跑在显卡上
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0", 
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 清理显存
print("🧹 正在清理临时显存...")
gc.collect()
torch.cuda.empty_cache()

print(f"✅ 加载完成！当前显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ================= 定义流式输出器 =================
# 它的作用是：生成一个字，就打印一个字，不用等全部生成完
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

messages = []
print("💡 系统提示: 输入 'exit' 退出，输入 'clear' 清空历史。")

while True:
    try:
        user_input = input("\n👤 User: ").strip()
    except EOFError:
        break

    if not user_input: continue
    if user_input.lower() in ['exit', 'quit', 'q']: break
    if user_input.lower() == 'clear':
        messages = []
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 历史已清空")
        continue

    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("🤖 Assistant: ", end="", flush=True) # 打印个头，后面接着流式输出
    
    # 开始推理
    # 注意：这里把生成的 id 也不要了，因为 streamer 会自动打印到屏幕上
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        streamer=streamer, # <--- 关键：把结果交给 streamer 处理
        temperature=0.7,
        top_p=0.9
    )

    # 为了保持历史记录，我们需要把生成的内容拿回来存进 messages
    # 这里的逻辑稍显复杂，是为了从 output 里提取出纯回复部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    messages.append({"role": "assistant", "content": response})