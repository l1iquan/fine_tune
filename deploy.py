from modelscope import AutoTokenizer
from modelscope import AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc

# ================= 配置区域 =================
model_name = "Qwen/Qwen3-14B" # 这里的路径保持你原来的
# ===========================================

print(f"🚀 正在加载 Tokenizer: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 1. 改为 8-bit 量化配置
# 8-bit 不需要 nf4 等参数，只需要 load_in_8bit=True
print("⚙️  正在配置 8-bit 量化模式...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

print("📥 正在加载模型到 GPU (这可能需要几分钟)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 2. 关键步骤：加载完后立即清理显存
# 这会把加载过程中产生的临时碎片清理掉，腾出空间给推理
print("🧹 正在清理加载产生的临时显存...")
gc.collect()
torch.cuda.empty_cache()

# 打印当前显存状态
mem_alloc = torch.cuda.memory_allocated() / 1024**3
mem_reserved = torch.cuda.memory_reserved() / 1024**3
print(f"✅ 模型加载完成！")
print(f"📊 当前显存实际占用: {mem_alloc:.2f} GB")
print(f"📊 当前显存预留总量: {mem_reserved:.2f} GB")
print("-" * 50)

# ================= 循环对话逻辑 =================
# 用于存储历史对话，实现“多轮对话”
messages = []

print("💡 系统提示: 输入 'exit' 或 'q' 退出对话，输入 'clear' 清空历史记录。")

while True:
    # 获取用户输入
    try:
        user_input = input("\n👤 User: ").strip()
    except EOFError:
        break

    if not user_input:
        continue
    
    # 退出命令
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("👋 再见！")
        break
        
    # 清空历史命令
    if user_input.lower() == 'clear':
        messages = []
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 历史记录已清空，显存已整理。")
        continue

    # 将用户输入加入历史
    messages.append({"role": "user", "content": user_input})

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. 推理生成
    # 8-bit 显存比 4-bit 紧张，所以 max_new_tokens 不要设得太疯狂，2048 足够日常对话
    # 如果显存爆了，尝试调小 max_new_tokens 或定期输入 clear
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048,  
            temperature=0.7,
            top_p=0.9
        )
    except torch.cuda.OutOfMemoryError:
        print("❌ 显存不足 (OOM)！正在自动清理并重置对话...")
        gc.collect()
        torch.cuda.empty_cache()
        messages = [] # 显存爆了通常只能清空历史
        continue

    # 获取纯粹的新生成内容（去掉输入的 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 打印回答
    print(f"🤖 Assistant: {response}")

    # 将 AI 的回答也加入历史，形成上下文
    messages.append({"role": "assistant", "content": response})