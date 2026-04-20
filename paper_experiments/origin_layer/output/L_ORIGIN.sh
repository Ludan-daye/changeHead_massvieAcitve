# === 来自 determine_origin_layer.py 的自动产出 ===
# 来源: results/ALL_EXPERIMENTS_SUMMARY_v2.json 的 exp2c
# 更新时: 替换 run_rq345_origin_layer.sh 中原 L_ORIGIN 数组

# 单层实验用的起源层 (RQ3 / RQ4 / RQ5-single)
declare -A L_ORIGIN=(
    [bloom_7b1]=3    # CONCENTRATED
    # [deepseek_v2_lite]=??  # ?, 无 exp2c 数据
    [falcon_7b]=3    # FEW-SOURCE
    [glm4_32b]=0    # CONCENTRATED
    [glm4_9b]=1    # FEW-SOURCE
    [gpt2]=3    # FEW-SOURCE
    [gptj_6b]=2    # CONCENTRATED
    [llama2_13b]=0    # FEW-SOURCE
    # [llama2_7b_chat]=??  # ?, 无 exp2c 数据
    [llama3.1_8b]=1    # FEW-SOURCE
    [mistral_7b_v03]=0    # CONCENTRATED
    [opt_6.7b]=1    # ANOMALY_NO_MLP_RESPONSE
    [qwen1.5_14b]=35   # DISPERSED
    [qwen2.5_0.5b]=0    # CONCENTRATED
    # [qwen2.5_0.5b_optimized]=??  # ?, 无 exp2c 数据
    [qwen2.5_7b]=3    # CONCENTRATED
    # [qwen2.5_7b_old_nan]=??  # ?, 无 exp2c 数据
    [qwen2_7b]=3    # CONCENTRATED
    [qwen3.5_27b]=54   # DISPERSED
    [qwen3.5_35b_a3b]=9    # FEW-SOURCE
    [qwen3.5_9b]=22   # DISPERSED
    [qwen3_0.6b]=2    # CONCENTRATED
    [qwen3_1.7b]=2    # FEW-SOURCE
    [qwen3_14b]=6    # DISPERSED
    [qwen3_30b_a3b]=1    # DISPERSED
    [qwen3_32b]=6    # DISPERSED
    [qwen3_4b]=6    # FEW-SOURCE
    [qwen3_8b]=6    # DISPERSED
    [yi_9b]=8    # DISPERSED
)

