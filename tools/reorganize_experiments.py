#!/usr/bin/env python3
"""
重组实验目录结构：从 models/{model}/{exp} 到 experiments/{exp}/{model}
"""
import os
import shutil
from pathlib import Path
import json

# 当前工作目录
BASE_DIR = Path("PROJECT_ROOT/results")
MODELS_DIR = BASE_DIR / "models"
NEW_EXPERIMENTS_DIR = BASE_DIR / "experiments"

# 实验名称映射（旧名 -> 新名）
EXP_MAPPING = {
    "exp1": "exp1",
    "exp1_feasibility_test": "exp1",
    "exp1_llama2_13b": "exp1",
    "exp1_llama2_7b_chat": "exp1",
    "exp1_opt_6.7b": "exp1",

    "exp2": "exp1a",  # Attention头贡献度 -> exp1a
    "exp2_llama2_13b": "exp1a",
    "exp2_opt_6.7b": "exp1a",

    "exp2b_mlp_layer_ablation": "exp2",  # MLP层贡献度 -> exp2

    # 跳过特殊测试实验（带字母后缀）
    "exp2a_mlp_feasibility_test": None,  # gpt2特殊测试，跳过
    "exp2c_mlp_internal": None,          # gpt2特殊测试，跳过

    "exp3_svd_alignment": "exp3",
    "exp3_opt_fire_test": "exp3",

    "exp4": "exp4",
    "exp4_opt_svd": "exp4",

    "exp4b": "exp4b",
    "exp4b_layer3_attention": "exp4b",
    "exp4b_layer31": "exp4b",

    # 跳过llama2特殊测试
    "exp4c_layer3_head_output": None,  # llama2特殊测试，跳过
    "exp4d_layer3_mlp": None,          # llama2特殊测试，跳过

    "exp6": "exp6",
}

def get_all_model_exp_pairs():
    """扫描所有模型的实验目录"""
    pairs = []

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # 跳过misc目录
        if model_name == "misc":
            continue

        # 找到所有exp*目录
        for exp_dir in model_dir.glob("exp*"):
            if exp_dir.is_dir():
                old_exp_name = exp_dir.name

                # 映射到新的实验名
                new_exp_name = EXP_MAPPING.get(old_exp_name, old_exp_name)

                # 跳过映射为None的实验（特殊测试）
                if new_exp_name is None:
                    continue

                pairs.append({
                    "model": model_name,
                    "old_exp": old_exp_name,
                    "new_exp": new_exp_name,
                    "old_path": exp_dir,
                    "new_path": NEW_EXPERIMENTS_DIR / new_exp_name / model_name
                })

    return pairs

def create_directory_structure(pairs):
    """创建新的目录结构"""
    print("📁 创建新目录结构...")

    exp_dirs = set()
    for pair in pairs:
        exp_dirs.add(pair["new_exp"])

    for exp_name in sorted(exp_dirs):
        exp_dir = NEW_EXPERIMENTS_DIR / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {exp_dir}")

    print(f"\n✅ 创建了 {len(exp_dirs)} 个实验目录")

def preview_reorganization(pairs):
    """预览重组计划"""
    print("\n" + "="*80)
    print("📋 重组计划预览")
    print("="*80)

    # 按新实验名分组
    by_exp = {}
    for pair in pairs:
        exp = pair["new_exp"]
        if exp not in by_exp:
            by_exp[exp] = []
        by_exp[exp].append(pair)

    for exp_name in sorted(by_exp.keys()):
        models = by_exp[exp_name]
        print(f"\n📊 {exp_name}/ ({len(models)} 个模型)")
        for pair in sorted(models, key=lambda x: x["model"]):
            old_rel = pair["old_path"].relative_to(MODELS_DIR)
            print(f"  {pair['model']:20s} <- models/{old_rel}")

    print("\n" + "="*80)
    print(f"总计: {len(pairs)} 个目录需要重组")
    print("="*80)

def copy_with_verification(pairs, dry_run=True):
    """复制数据并验证"""
    print("\n" + "="*80)
    if dry_run:
        print("🔍 DRY RUN 模式 (不会实际复制)")
    else:
        print("🚀 开始复制数据...")
    print("="*80)

    success = 0
    failed = 0

    for i, pair in enumerate(pairs, 1):
        old_path = pair["old_path"]
        new_path = pair["new_path"]

        print(f"\n[{i}/{len(pairs)}] {pair['model']} / {pair['new_exp']}")
        print(f"  源: {old_path}")
        print(f"  目标: {new_path}")

        if dry_run:
            # 仅检查源目录是否存在
            if old_path.exists():
                file_count = sum(1 for _ in old_path.rglob("*") if _.is_file())
                size_mb = sum(f.stat().st_size for f in old_path.rglob("*") if f.is_file()) / 1024 / 1024
                print(f"  ✅ 源存在: {file_count} 个文件, {size_mb:.1f} MB")
                success += 1
            else:
                print(f"  ❌ 源不存在")
                failed += 1
        else:
            # 实际复制
            try:
                if new_path.exists():
                    print(f"  ⚠️  目标已存在，跳过")
                    continue

                shutil.copytree(old_path, new_path)

                # 验证
                old_files = set(f.relative_to(old_path) for f in old_path.rglob("*") if f.is_file())
                new_files = set(f.relative_to(new_path) for f in new_path.rglob("*") if f.is_file())

                if old_files == new_files:
                    print(f"  ✅ 复制成功: {len(new_files)} 个文件")
                    success += 1
                else:
                    print(f"  ⚠️  文件数不匹配: {len(old_files)} -> {len(new_files)}")
                    failed += 1

            except Exception as e:
                print(f"  ❌ 失败: {e}")
                failed += 1

    print("\n" + "="*80)
    print(f"✅ 成功: {success}")
    print(f"❌ 失败: {failed}")
    print("="*80)

    return success, failed

def generate_report(pairs):
    """生成重组报告"""
    report = {
        "timestamp": "2025-12-19 15:45",
        "total_moves": len(pairs),
        "by_experiment": {},
        "by_model": {}
    }

    # 按实验统计
    for pair in pairs:
        exp = pair["new_exp"]
        if exp not in report["by_experiment"]:
            report["by_experiment"][exp] = []
        report["by_experiment"][exp].append(pair["model"])

    # 按模型统计
    for pair in pairs:
        model = pair["model"]
        if model not in report["by_model"]:
            report["by_model"][model] = []
        report["by_model"][model].append(pair["new_exp"])

    report_path = NEW_EXPERIMENTS_DIR / "reorganization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 报告已保存: {report_path}")

def main():
    import sys

    print("="*80)
    print("🔄 实验目录重组工具")
    print("="*80)
    print(f"源目录: {MODELS_DIR}")
    print(f"目标目录: {NEW_EXPERIMENTS_DIR}")
    print("="*80)

    # 扫描
    print("\n🔍 扫描所有实验目录...")
    pairs = get_all_model_exp_pairs()
    print(f"✅ 找到 {len(pairs)} 个实验目录")

    # 预览
    preview_reorganization(pairs)

    # 询问是否继续
    print("\n❓ 请选择操作:")
    print("  1. DRY RUN (仅检查，不复制)")
    print("  2. 开始复制")
    print("  3. 退出")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请输入选择 [1/2/3]: ").strip()

    if choice == "1":
        # DRY RUN
        create_directory_structure(pairs)
        copy_with_verification(pairs, dry_run=True)
        generate_report(pairs)
    elif choice == "2":
        # 实际复制
        confirm = input("⚠️  确认开始复制? 这将复制大量数据。输入 'YES' 确认: ").strip()
        if confirm == "YES":
            create_directory_structure(pairs)
            success, failed = copy_with_verification(pairs, dry_run=False)
            generate_report(pairs)

            if failed == 0:
                print("\n🎉 重组完成！所有数据已成功复制。")
            else:
                print(f"\n⚠️  重组完成，但有 {failed} 个失败。")
        else:
            print("❌ 已取消")
    else:
        print("👋 已退出")

if __name__ == "__main__":
    main()
