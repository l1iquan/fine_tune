import torch
from threading import Thread
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)
from peft import PeftModel
import gc
import os

# ================= 配置区域 =================
model_path = r"F:\huggingface\models\Qwen\Qwen3-14B" 
lora_path = "output_final" # 你的微调结果路径
# ===========================================

# 1. 显存大扫除
gc.collect()
torch.cuda.empty_cache()

# 2. 4-bit 量化配置
print("⚙️  正在配置 4-bit 量化...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"🚀 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"🚀 正在加载模型 (强制限制显存用法)...")

# === 【核心修复】 ===
# 我们给显卡设一个“软上限”：16GB。
# 4-bit 模型本身约 9.5GB。
# 设置 16GB，既保证模型能装进去，又强制阻止它一开始就占满 24GB。
# 这样加载完后，你应该会看到显存占用在 10GB - 11GB 左右。
# 剩下的 13GB 显存，才是留给流式对话和长上下文用的！
max_memory_map = {0: "16GB", "cpu": "99GB"}

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory=max_memory_map  # <--- 加上这行救命代码
)

# 3. 挂载 LoRA
print(f"🔄 正在挂载 LoRA: {lora_path}")
model = PeftModel.from_pretrained(base_model, lora_path)

# 打印一下真实的显存占用
print(f"✅ 加载完成！当前显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
# === 新增：强制释放 PyTorch 的缓存空房 ===
print("🧹 正在清理加载阶段产生的缓存碎片...")
gc.collect()
torch.cuda.empty_cache() # <--- 这行命令会把那 12GB 空房还给 Windows
# ========================================

# 再打印一次，你会发现任务管理器的数值降下来了
print(f"📉 清理后显存状态：")
print(f"   - 实际模型占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"   - 总显存占用 (含缓存): {torch.cuda.memory_reserved()/1024**3:.2f} GB")

print("-" * 60)

# ================= 连续对话主循环 =================
history = [] 

print("💡 输入内容开始对话。指令：'clear' 清空，'exit' 退出。")

while True:
    try:
        user_input = input("\n👤 User: ").strip()
    except EOFError:
        break

    if not user_input: continue
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        break
    
    if user_input.lower() == 'clear':
        history = []
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 记忆已清空")
        continue

    # 构建 Prompt
    history.append({"role": "user", "content": user_input})
    
    input_str = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([input_str], return_tensors="pt").to(model.device)

    # 流式输出配置
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=512, # 单次回复最大长度
        temperature=0.7,
        do_sample=True
    )

    # 启动生成线程
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 打印输出
    print("🤖 Assistant: ", end="", flush=True)
    full_response = ""
    
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_response += new_text
    
    print() 

    # 记录历史
    history.append({"role": "assistant", "content": full_response})
    
    # 简单的显存保护：只保留最近10轮对话，防止历史太长爆显存
    if len(history) > 20:
        history = history[-20:]