#!/bin/bash
# 隐私信息清理脚本
# 用于匿名化提交前清理所有个人信息

set -e

echo "开始清理隐私信息..."

# 备份当前目录
BACKUP_DIR="./privacy_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "创建备份目录: $BACKUP_DIR"

# 1. 清理Python文件中的硬编码路径
echo "正在清理Python文件中的路径..."
find . -name "*.py" -type f ! -path "./.git/*" ! -path "./privacy_backup*" -exec sed -i.bak \
    -e 's|PROJECT_ROOT|PROJECT_ROOT|g' \
    -e 's|PROJECT_ROOT|PROJECT_ROOT|g' \
    -e 's|/home/vicuna/ludan/models|LOCAL_MODELS_DIR|g' \
    -e 's|/home/ludandaye/reaserch/massvieActive/massive-activations|PROJECT_ROOT|g' \
    {} \;

# 2. 清理Markdown文档中的路径
echo "正在清理Markdown文档中的路径..."
find . -name "*.md" -type f ! -path "./.git/*" ! -path "./privacy_backup*" -exec sed -i.bak \
    -e 's|/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive|PROJECT_ROOT|g' \
    -e 's|/home/vicuna/ludan/massActive|PROJECT_ROOT|g' \
    -e 's|/home/ludandaye/reaserch/massvieActive/massive-activations|PROJECT_ROOT|g' \
    -e 's|\\\\wsl.localhost\\\\Ubuntu\\\\home\\\\ludandaye\\\\reaserch\\\\massvieActive\\\\massive-activations|PROJECT_ROOT|g' \
    -e 's|ludandaye|RESEARCHER|g' \
    -e 's|Ludandaye|RESEARCHER|g' \
    -e 's|vicuna|USER|g' \
    -e 's|ludan|RESEARCHER|g' \
    {} \;

# 3. 清理Shell脚本中的路径
echo "正在清理Shell脚本中的路径..."
find . -name "*.sh" -type f ! -path "./.git/*" ! -path "./privacy_backup*" -exec sed -i.bak \
    -e 's|PROJECT_ROOT|PROJECT_ROOT|g' \
    -e 's|PROJECT_ROOT|PROJECT_ROOT|g' \
    {} \;

# 4. 清理JSON配置文件中的路径（如果有）
echo "正在清理JSON文件中的路径..."
find . -name "*.json" -type f ! -path "./.git/*" ! -path "./privacy_backup*" ! -path "./model_weights/*" -exec sed -i.bak \
    -e 's|/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive|PROJECT_ROOT|g' \
    -e 's|/home/vicuna/ludan|PROJECT_ROOT|g' \
    -e 's|/home/ludandaye|PROJECT_ROOT|g' \
    {} \;

# 5. 清理TXT文件中的路径
echo "正在清理TXT文件中的路径..."
find . -name "*.txt" -type f ! -path "./.git/*" ! -path "./privacy_backup*" -exec sed -i.bak \
    -e 's|/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive|PROJECT_ROOT|g' \
    -e 's|/home/vicuna/ludan|PROJECT_ROOT|g' \
    -e 's|/home/ludandaye/reaserch/massvieActive/massive-activations|PROJECT_ROOT|g' \
    {} \;

# 6. 移除所有备份文件
echo "正在清理备份文件..."
find . -name "*.bak" -type f ! -path "./privacy_backup*" -delete

# 7. 检查是否还有残留的隐私信息
echo ""
echo "检查残留的隐私信息..."
REMAINING=$(grep -r "vicuna\|ludandaye\|Ludandaye\|ludan\|d5f4cfb6-8afe-40a4-8650-2965046cd208" --include="*.py" --include="*.md" --include="*.sh" --include="*.txt" . 2>/dev/null | grep -v ".git" | grep -v "privacy_cleanup.sh" | grep -v "Binary" || true)

if [ -z "$REMAINING" ]; then
    echo "✅ 未发现残留的隐私信息"
else
    echo "⚠️  仍有以下文件包含隐私信息:"
    echo "$REMAINING"
    echo ""
    echo "请手动检查这些文件"
fi

echo ""
echo "清理完成！"
echo ""
echo "下一步建议:"
echo "1. 检查 git config user.name 和 user.email"
echo "2. 如需匿名提交，运行: git config user.name 'Anonymous' && git config user.email 'anonymous@example.com'"
echo "3. 检查生成的差异: git diff"
echo "4. 如果确认无误，提交更改: git add -A && git commit -m 'Remove privacy information'"
