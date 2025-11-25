import os
import torch
from swift.llm import TrainArguments, sft_main

# 1. 减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 2. 显存优化配置 - 最激进的方案
sft_args = TrainArguments(
    # === 模型与路径 ===
    model_type='qwen3',
    model=r'F:\huggingface\models\Qwen\Qwen3-14B',
    
    dataset=['train.jsonl'],
    
    # === 【关键修改】使用更低的精度和强制CPU ===
    device_map='auto', 
    
    # === 【关键修改】使用更激进的量化配置 ===
    model_kwargs={
        "low_cpu_mem_usage": True,
        # 限制显卡只用 10GB，强制更多层到内存
        "max_memory": {0: "10GB", "cpu": "99GB"},
        # 使用更低的精度
        "torch_dtype": torch.float16,
        # 强制使用更激进的量化
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16
        }
    },
    
    # === 显存优化 ===
    quant_bits=4,                # 4bit 量化
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16, 
    
    # === LoRA 配置 ===
    train_type='lora',
    lora_rank=2,              # 进一步降低rank
    lora_alpha=8,
    
    # === 训练参数 ===
    num_train_epochs=10,
    learning_rate=1e-4,
    
    output_dir='output',
    
    # 【保险】进一步缩短长度
    max_length=128,
    
    gradient_checkpointing=True,
    save_steps=50,
    
    # 使用更低的精度
    bf16=False,
    fp16=True,
    
    # 禁用一些内存消耗大的功能
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

if __name__ == '__main__':
    print("🚀 开始训练...")
    print("   注意：这次使用了最激进的显存优化策略...")
    
    try:
        result = sft_main(sft_args)
        print(f"🎉 训练完成！权重保存在: {result['best_model_checkpoint']}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        # 如果还是报错，说明需要更极端的方案
        if "cuda:0" in str(e) or "OOM" in str(e):
             print("\n👉 建议：如果这次还不行，请考虑：")
             print("1. 使用更小的模型 (如 Qwen3-7B)")
             print("2. 使用 DeepSpeed ZeRO 优化")
             print("3. 使用 CPU 训练")
             print("4. 降级到 Swift 2.4.2: pip install ms-swift==2.4.2")
