# `fixes/lib/` — model_utils.py 修复

## 修了什么

**Bug B3**：`get_mlp_submodules(model_name, layer)` 原本不识别 `glm4` / `yi` 家族，会抛：

```
ValueError: Cannot identify MLP submodules for model 'glm4_9b'. Add support in model_utils.py.
```

影响：glm4_9b, glm4_32b, yi_9b 在 RQ3 脚本里**直接 crash**，不产出 json。

## 修复方法

在已有的 SwiGLU 分支（原本覆盖 llama / mistral / qwen）里加上 `glm4` 和 `yi`：

```python
if ("llama" in model_name or "mistral" in model_name or "qwen" in model_name
        or "glm4" in model_name or "yi" in model_name):
    return {
        'up_proj': layer.mlp.up_proj,
        ...
    }
```

**原理**：glm4/yi 和 llama 都用 SwiGLU，HF 里的 MLP 属性名完全一致（`up_proj / gate_proj / down_proj / act_fn`），可以直接复用模板。

## 部署

把本目录下的 `model_utils.py` **覆盖**到原位置：

```bash
cd <repo-root>
cp paper_experiments/fixes/lib/model_utils.py paper_experiments/lib/model_utils.py
```

## 验证

没有独立运行入口（这是个库文件，由 RQ3 等脚本 import）。

```bash
# 方法 1：跑仓库根的 sentinel_test.sh（自动测试 Test A）
bash paper_experiments/fixes/sentinel_test.sh

# 方法 2：手动验证
python3 -c "
import sys; sys.path.insert(0, 'paper_experiments')
from lib.model_utils import get_mlp_submodules
# glm4_9b 模拟调用，不应该再抛 ValueError
"
```

**预期**：6 个 sentinel 模型（含 glm4_9b / yi_9b）全部返回 dict，不抛异常。

## 不变的部分

- 原有 12 个模型家族（llama / mistral / qwen / gpt2 / gptj / bloom / falcon / opt / phi / pythia 等）的逻辑**完全不动**
- 其他函数（`enable_custom_block` 等）不动

## 影响的下游脚本

- `paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py`（主要消费者）
- 任何其他 `from lib.model_utils import get_mlp_submodules` 的脚本
