# `sentinel_test.sh` — 修复验证脚本

## 做什么

本地快速验证所有 6 个 bug 修复**没有回归**，**不用加载任何模型**（纯逻辑检查）。

6 个测试点覆盖 6 个 bug 修复：

| Test | 验证的 Bug | 检查点 |
|:-:|:-:|---|
| A | B3 | `get_mlp_submodules` 能处理 glm4/yi/qwen1.5/qwen3.5 + 已有模型 |
| B | B7 | MLPDisableHook 对 dense tensor 和 MoE tuple 都正确置零 |
| C | B4 | RQ6 `get_mlp_down_proj` 能处理 glm4/yi |
| D | B5 | `get_critical_layer` 正确读 L_ORIGIN.json；env override 生效；未知模型 raise |
| E | B6 | `run_and_collect_ma` 扫所有层、记录 peak_layer / ablation_layer；MoE guard；`--layer_id` 参数 |
| F | B1 | `FunctionWordSVDTracker` 正确分类 func / struct / content 三类 |

## 怎么运行

```bash
cd <repo-root>
bash paper_experiments/fixes/sentinel_test.sh
```

**要求**：
- Python 3 可用（`python3` 命令）
- `paper_experiments/origin_layer/output/L_ORIGIN.json` 存在（Test D 需要）
- Bash 标准工具（`cd`, `echo`, `date`）

**不要求**：
- 不需要 GPU / CUDA
- 不需要 HuggingFace 模型权重
- 不需要 transformers / torch 的完整安装（虽然 torch 用作 tensor 工具）

## 预期输出

```
Sentinel test output → /tmp/fix_sentinel_<timestamp>

Test A: B3 white-list (get_mlp_submodules with fake layer)
  ✓ B3: all 7 model names resolved

Test B: B7 MoE tuple hook
  ✓ B7: dense + MoE tuple both handled

Test C: B4 RQ6 get_mlp_down_proj
  ✓ B4: all 8 model names resolved

Test D: B5 get_critical_layer reads L_ORIGIN.json
  ✓ B5: 4 real models + env override + fail-fast

Test E: B6 run_and_collect_ma pattern check
  ✓ B6: all structural markers present

Test F: B1 FunctionWordSVDTracker classification
  ✓ B1: func/struct/content all tracked correctly

===================================
Sentinel results: 6 passed, 0 failed
===================================
All checks passed. Safe to deploy.
```

**退出码**：全过 → `0`；任一失败 → `1`（可用在 CI）

## 失败时怎么办

如果某个 Test 失败，对应的 bug 修复可能损坏：

| 失败 Test | 检查方向 |
|:-:|---|
| A | `fixes/lib/model_utils.py` 第 229 行的 SwiGLU 分支是否包含 glm4/yi |
| B | `fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` 的 `MLPDisableHook.__call__` 是否处理 tuple |
| C | `fixes/RQ6_v_ablation/exp6_v_ablation.py` 的 `get_mlp_down_proj` 是否包含 glm4/yi 分支 |
| D | `get_critical_layer` 的 L_ORIGIN.json 路径解析；确认 `paper_experiments/origin_layer/output/L_ORIGIN.json` 存在；环境变量/fail-fast 分支 |
| E | `run_and_collect_ma` 是否扫全层；MoE guard；`--layer_id` argparse |
| F | `FunctionWordSVDTracker.add_token` 是否存所有 token（不只是功能词）；`is_structural_token` 分类 |

## 局限

- **不加载真实模型** —— 用 FakeLayer / FakeMLP mock 对象
- **不测 forward pass** —— 只测静态代码逻辑（模块 import + 函数调用）
- **不验证数值正确性** —— 只检查"不 crash + 返回类型对"

**真正的数值验证**要在部署后跑一次 gpt2 + glm4_9b 的 5-sample mini run，对比 baseline 和 RQ2a 量级。见各目录 README 的 "验证修复" 章节。

## CI 集成（可选）

```yaml
# .github/workflows/sentinel.yml
name: Fix Sentinel
on: [push, pull_request]
jobs:
  sentinel:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with: {submodules: true}
      - uses: actions/setup-python@v4
        with: {python-version: '3.10'}
      - run: pip install torch numpy
      - run: bash paper_experiments/fixes/sentinel_test.sh
```

## 临时文件

脚本在 `/tmp/fix_sentinel_<timestamp>` 下创建工作目录。**当前空的**（所有测试都用 Fake 对象），实际部署后可用这个目录存 mini-run 验证输出。
