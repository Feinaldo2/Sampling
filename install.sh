# conda create -n slow_fast_sampling python=3.12
# conda activate slow-fast-sampling
# pip install -r requirements.txt
# huggingface-cli download --resume-download GSAI-ML/LLaDA-8B-Base
# huggingface-cli download --resume-download Dream-org/Dream-v0-Base-7B

repo 根目录执行：把自定义文件覆盖到 HF 缓存里的 Dream/generation_utils.py
CUSTOM_FILE="sampling_utils/dream_generation_utils.py"
HF_HOME="/home/zhaoyifei/.cache/huggingface"
TARGET_FILE=$(find "/home/zhaoyifei/Sampling/Models/Dream-base" \
                -type f -name generation_utils.py | head -n 1)

if [ -z "$TARGET_FILE" ]; then
  echo "❌ 未找到 generation_utils.py，确认已用 transformers 加载过 Dream 模型" >&2
  exit 1
fi

echo "🔄 正在替换 $TARGET_FILE"
cp "$CUSTOM_FILE" "$TARGET_FILE"
echo "✅ 替换完成！"