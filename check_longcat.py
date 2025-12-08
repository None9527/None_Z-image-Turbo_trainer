"""
检查 LongCat 模型文件完整性
"""
from pathlib import Path
import json

model_dir = Path("d:/AI/None_Z-image-Turbo_trainer/longcat_models")

print("=" * 60)
print("LongCat 模型文件检查")
print("=" * 60)

# 检查各组件
components = {
    "transformer": ["config.json", "diffusion_pytorch_model.safetensors.index.json"],
    "vae": ["config.json", "diffusion_pytorch_model.safetensors"],
    "text_encoder": ["config.json", "model.safetensors.index.json"],
    "tokenizer": ["tokenizer.json", "tokenizer_config.json"],
    "scheduler": ["scheduler_config.json"]
}

total_size = 0

for comp_name, expected_files in components.items():
    comp_path = model_dir / comp_name
    print(f"\n[{comp_name}]")
    
    if not comp_path.exists():
        print(f"  ✗ 目录不存在")
        continue
    
    print(f"  ✓ 目录存在")
    
    # 检查关键文件
    for file_name in expected_files:
        file_path = comp_path / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            print(f"  ✓ {file_name}: {size / (1024**3):.2f} GB")
        else:
            print(f"  ✗ {file_name}: 缺失")
    
    # 检查分片文件
    if comp_name in ["transformer", "text_encoder"]:
        index_file = comp_path / (expected_files[1] if len(expected_files) > 1 else "model.safetensors.index.json")
        if index_file.exists():
            try:
                with open(index_file) as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                shard_files = set(weight_map.values())
                
                print(f"  分片文件: {len(shard_files)} 个")
                missing = []
                for shard in shard_files:
                    shard_path = comp_path / shard
                    if shard_path.exists():
                        size = shard_path.stat().st_size
                        total_size += size
                    else:
                        missing.append(shard)
                
                if missing:
                    print(f"  ✗ 缺失分片: {len(missing)} 个")
                    for m in missing[:3]:
                        print(f"    - {m}")
                else:
                    print(f"  ✓ 所有分片完整")
            except Exception as e:
                print(f"  ✗ 读取索引失败: {e}")

print("\n" + "=" * 60)
print(f"总大小: {total_size / (1024**3):.2f} GB / ~35 GB")
print(f"完成度: {total_size / (35 * 1024**3) * 100:.1f}%")
print("=" * 60)
