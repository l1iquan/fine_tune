import torch
import os
import gc # 引入垃圾回收
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset

# 1. 基础配置
model_path = r"F:\huggingface\models\Qwen\Qwen3-14B"
data_file = "train.jsonl"
output_dir = "output_final"

# 显存清理：开始前先大扫除
gc.collect()
torch.cuda.empty_cache()

# 2. 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("🚀 1. 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🚀 2. 正在加载模型 (流式量化加载)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"✅ 模型加载完成。当前显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# 3. 准备 LoRA 环境
print("⚙️  3. 准备 LoRA 环境...")
# 开启梯度检查点：这是省显存的关键，确保它开启！
model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 数据处理
print("📚 4. 处理数据...")
def process_func(example):
    # =========== 核心修改 ===========
    # 从 1024 改为 512。
    # 14B 模型 + 24G 显存 + Windows，512 是安全线。
    # 长度减半，训练时的动态显存占用会减少约 40%-50%。
    MAX_LENGTH = 512 
    # ===============================
    
    instruction = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["query"]}],
        add_generation_prompt=True,
        tokenize=False
    )
    response = example["response"] + tokenizer.eos_token
    
    instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    
    input_ids = instruction_ids + response_ids
    labels = [-100] * len(instruction_ids) + response_ids
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids), "labels": labels}

dataset = load_dataset("json", data_files=data_file, split="train")
tokenized_dataset = dataset.map(process_func, remove_columns=dataset.column_names)

# 5. 训练器配置
print("🔥 5. 开始训练...")
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16, 
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=1,            # 每一步都打印，让你看到进度
        fp16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        report_to="none"
    ),
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

model.config.use_cache = False
trainer.train()
trainer.save_model(output_dir)
print(f"🎉 训练结束，模型保存在 {output_dir}")