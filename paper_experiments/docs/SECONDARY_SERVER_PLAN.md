# Secondary Server Execution Plan — 6 HF-Download Models

> **Date**: 2026-04-21
> **Target**: Run full RQ1/2a/2c/3/4/5s/5m/6 sweep on a **second server** for the 6 dense
> models whose weights are NOT local on the primary server (117.50.223.194).
> **Primary server** is concurrently running Stage-2 RQ3 batch for the other 15 dense models.
> **Merge strategy**: output JSON paths are identical on both servers; post-run rsync
> unifies into `results/wikitext_run_2026_04_21/`.
>
> **Read first**:
> - `paper_experiments/docs/DEBUG_SUMMARY.md` — what the 6 fixes do
> - `paper_experiments/fixes/README.md` — file replacement table
> - `paper_experiments/fixes/sentinel_test.md` — how the verification script works

---

## Target models and L_ORIGIN

| Model | Size | HF ID | L_ORIGIN |
|---|:-:|---|:-:|
| `bloom_7b1`      | ~15 GB | `bigscience/bloom-7b1`          | 3 |
| `falcon_7b`      | ~15 GB | `tiiuae/falcon-7b`              | 3 |
| `gptj_6b`        | ~24 GB (fp32 on disk) | `EleutherAI/gpt-j-6b` | 2 |
| `llama2_13b`     | ~28 GB | `meta-llama/Llama-2-13b-hf`     | 0 |
| `mistral_7b_v03` | ~15 GB | `mistralai/Mistral-7B-v0.3`     | 0 |
| `opt_6.7b`       | ~14 GB | `facebook/opt-6.7b`             | 1 |

Disk budget: ~110 GB for weights + ~5 GB for results → **reserve 150 GB**.
GPU budget: single A800 80 GB fits all 6 one-at-a-time. `llama2_13b` @ fp16 ≈ 26 GB loaded;
RQ3/RQ4 overhead ≈ +30 GB → should fit with ~20 GB headroom.

---

## Phase 0 — Pre-flight (≈20 min)

### Task 0.1 — Verify Python env (2 min)

```bash
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python3 -c "import transformers; print('transformers', transformers.__version__)"
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0), 'mem:', torch.cuda.get_device_properties(0).total_memory / 2**30, 'GB')"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
```

**Pass criteria**:
- Python ≥ 3.10
- torch ≥ 2.x with CUDA
- transformers **must be 5.x** (same as primary). If it's 4.x, see Recovery §R3 — the `modify_llama.py` fix (B8) targets the 5.x API.
- GPU ≥ 75 GB free

### Task 0.2 — Verify disk (1 min)

```bash
df -h ~    # need ≥ 150 GB free on home / HF cache partition
```

If tight, export a bigger scratch partition as HF cache:
```bash
export HF_HOME=/data/hf_cache
mkdir -p $HF_HOME
```
Add to `~/.bashrc` if persistent.

### Task 0.3 — Clone / sync the repo (3 min)

If the repo isn't on the server yet:
```bash
cd ~
git clone <repo-url> ma
cd ~/ma
git log --oneline -5
```

If already there:
```bash
cd ~/ma
git fetch --all
git status
git pull
git log --oneline -5
```

**Pass criteria**: `paper_experiments/fixes/` exists with 4 `exp*.py` files + `lib/model_utils.py` + `monkey_patch/modify_llama.py` + `sentinel_test.sh`.

### Task 0.4 — HuggingFace access test (3 min)

```bash
cd ~/ma
python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
for repo in ["bigscience/bloom-7b1", "tiiuae/falcon-7b", "EleutherAI/gpt-j-6b",
             "meta-llama/Llama-2-13b-hf", "mistralai/Mistral-7B-v0.3", "facebook/opt-6.7b"]:
    try:
        info = api.model_info(repo)
        print(f"OK  {repo} — {info.sha[:10]}")
    except Exception as e:
        print(f"FAIL {repo}: {e}")
EOF
```

**Pass criteria**: all 6 print `OK …`.

If **Llama-2-13b** or **Mistral-7B-v0.3** fails with 401/403: both are gated. Accept the license on HuggingFace, create an access token at https://huggingface.co/settings/tokens, then:
```bash
huggingface-cli login
```
Re-run the check.

If no internet access → see Recovery §R1.

### Task 0.5 — Write the secondary model list (2 min)

```bash
cat > ~/ma/paper_experiments/fixes/models_secondary.txt <<'EOF'
# 6 dense models downloaded from HF on secondary server
# Sorted smallest-first (matches run_pipeline.py auto-sort)
opt_6.7b
gptj_6b
bloom_7b1
falcon_7b
mistral_7b_v03
llama2_13b
EOF
cat ~/ma/paper_experiments/fixes/models_secondary.txt
```

---

## Phase 1 — Deploy the 5 fix files (≈5 min)

### Task 1.1 — Back up originals (1 min)

```bash
cd ~/ma
BACKUP=paper_experiments/_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP/lib $BACKUP/RQ2_mlp_source $BACKUP/RQ3_function_words \
         $BACKUP/experiments/exp6_v_ablation $BACKUP/monkey_patch
cp paper_experiments/lib/model_utils.py                                    $BACKUP/lib/
cp paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py          $BACKUP/RQ2_mlp_source/
cp paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py $BACKUP/RQ3_function_words/
cp experiments/exp6_v_ablation/exp6_v_ablation.py $BACKUP/experiments/exp6_v_ablation/
cp paper_experiments/monkey_patch/modify_llama.py                          $BACKUP/monkey_patch/
echo "Backed up to $BACKUP"
```

### Task 1.2 — Deploy the 5 files (2 min)

```bash
cd ~/ma
cp paper_experiments/fixes/lib/model_utils.py \
   paper_experiments/lib/model_utils.py

cp paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
   paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py

cp paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py \
   paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py

cp paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py \
   experiments/exp6_v_ablation/exp6_v_ablation.py

cp paper_experiments/fixes/monkey_patch/modify_llama.py \
   paper_experiments/monkey_patch/modify_llama.py
```

### Task 1.3 — Verify each copy (1 min)

```bash
for pair in \
  "paper_experiments/fixes/lib/model_utils.py paper_experiments/lib/model_utils.py" \
  "paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py" \
  "paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py" \
  "paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py experiments/exp6_v_ablation/exp6_v_ablation.py" \
  "paper_experiments/fixes/monkey_patch/modify_llama.py paper_experiments/monkey_patch/modify_llama.py"
do
  set -- $pair
  if diff -q "$1" "$2" > /dev/null; then echo "OK  $2"; else echo "DIFF $2"; fi
done
```

**Pass criteria**: all 5 print `OK …`.

### Task 1.4 — Run sentinel test (2 min, CPU-only)

```bash
cd ~/ma
bash paper_experiments/fixes/sentinel_test.sh
```

**Pass criteria**: last line says `All checks passed. Safe to deploy.` — 6/6 tests pass.

**Do not proceed** until sentinel passes.

---

## Phase 2 — Warm HF cache (≈60–90 min, parallel-able)

### Task 2.1 — Start prefetch in tmux

```bash
cd ~/ma
tmux new -d -s hf_prefetch
tmux send-keys -t hf_prefetch "bash" Enter
tmux send-keys -t hf_prefetch "python3 - <<'PY'
from huggingface_hub import snapshot_download
models = [
    ('facebook/opt-6.7b',            'opt_6.7b'),
    ('EleutherAI/gpt-j-6b',          'gptj_6b'),
    ('bigscience/bloom-7b1',         'bloom_7b1'),
    ('tiiuae/falcon-7b',             'falcon_7b'),
    ('mistralai/Mistral-7B-v0.3',    'mistral_7b_v03'),
    ('meta-llama/Llama-2-13b-hf',    'llama2_13b'),
]
for hf_id, key in models:
    print(f'=== {key} ({hf_id}) ===', flush=True)
    snapshot_download(
        repo_id=hf_id,
        allow_patterns=['*.json', '*.txt', '*.model', 'tokenizer*',
                        '*.safetensors', '*.bin'],
    )
    print(f'done {key}', flush=True)
print('ALL DOWNLOADS COMPLETE', flush=True)
PY
" Enter
```

Check progress:
```bash
tmux attach -t hf_prefetch    # Ctrl-b d 退出
# or:
tmux capture-pane -p -t hf_prefetch | tail -20
```

### Task 2.2 — Wait for "ALL DOWNLOADS COMPLETE"

### Task 2.3 — Confirm cache sizes

```bash
du -sh ${HF_HOME:-~/.cache/huggingface}/hub/models--* | sort -h
```

**Pass criteria**: ~110 GB across 6 entries.

---

## Phase 3 — Dry run and smoke test (≈15 min)

### Task 3.1 — Pipeline dry-run

```bash
cd ~/ma
python3 paper_experiments/fixes/run_pipeline.py \
    --models_file paper_experiments/fixes/models_secondary.txt \
    --rqs RQ1 RQ2a RQ2c RQ3 RQ4 RQ5s RQ5m RQ6 \
    --nsamples 30 \
    --dry_run
```

**Pass criteria**:
- 6 × 8 = **48 lines** of `[N/48] DRY-RUN: …`
- No `SKIP (no L_ORIGIN)` for any of the 6 models
- No Python errors

### Task 3.2 — Mini smoke test

```bash
cd ~/ma
python3 paper_experiments/fixes/run_pipeline.py \
    --models opt_6.7b \
    --rqs RQ1 \
    --nsamples 5 \
    --timeout 1800
```

**Pass criteria**: `✓ ok` for `opt_6.7b RQ1`；输出目录 `paper_experiments/results/wikitext_run_2026_04_21/RQ1/opt_6.7b/` 有 `baseline/` + `all_heads_disabled/` 子目录。

If this fails → Recovery §R2。

---

## Phase 4 — Full experiment sweep (≈3–4 h)

### Task 4.1 — Launch full sweep in tmux

```bash
cd ~/ma
mkdir -p paper_experiments/logs
tmux new -d -s rqsweep
tmux send-keys -t rqsweep "bash" Enter
tmux send-keys -t rqsweep "cd ~/ma && python3 paper_experiments/fixes/run_pipeline.py \
    --models_file paper_experiments/fixes/models_secondary.txt \
    --rqs RQ1 RQ2a RQ2c RQ3 RQ4 RQ5s RQ5m RQ6 \
    --nsamples 30 \
    --timeout 3600 \
    --vram_threshold_gb 5.0 2>&1 | tee paper_experiments/logs/secondary_sweep.log" Enter
```

Detach (`Ctrl-b d`)，查询进度：
```bash
tmux capture-pane -p -t rqsweep | tail -30
tail -f ~/ma/paper_experiments/logs/secondary_sweep.log
```

Budget per phase:

| Phase | RQ | 6 models ETA |
|:-:|:-:|:-:|
| A | RQ1  | ~15 min |
| B | RQ2a | ~15 min |
| C | RQ2c | ~15 min |
| D | RQ3  | ~45 min |
| E | RQ4  | ~45 min |
| F | RQ5s | ~30 min |
| F | RQ5m | ~30 min |
| G | RQ6  | ~45 min |
| **Total** | | **≈3.5 h** |

### Task 4.2 — Watch for failures

```bash
watch -n 60 "jq '.results | group_by(.status) | map({status:.[0].status, n:length})' \
  ~/ma/paper_experiments/logs/pipeline_*/summary.json | tail -30"
```

### Task 4.3 — Confirm completion

```bash
LATEST=$(ls -td ~/ma/paper_experiments/logs/pipeline_* | head -1)
jq '{done, total, ok: ([.results[] | select(.status=="ok")] | length),
     failed: [.results[] | select(.status!="ok") | {model, rq, status, reason}]}' \
  $LATEST/summary.json
```

**Pass criteria**: `done == total == 48`, `ok == 48`, `failed == []`.

---

## Phase 5 — Result verification (≈10 min)

### Task 5.1 — All output dirs present

```bash
cd ~/ma/paper_experiments/results/wikitext_run_2026_04_21
for rq in RQ1 RQ2a RQ2c RQ3 RQ4 RQ5s RQ5m RQ6; do
  for m in opt_6.7b gptj_6b bloom_7b1 falcon_7b mistral_7b_v03 llama2_13b; do
    d=$rq/$m
    if [ -d "$d" ] && [ -n "$(ls -A $d 2>/dev/null)" ]; then
      echo "OK  $d ($(ls $d | wc -l) entries)"
    else
      echo "MISS $d"
    fi
  done
done
```

**Pass criteria**: all 48 print `OK …`. Zero `MISS`.

### Task 5.2 — RQ3 word_stats structure (B1 validation)

```bash
cd ~/ma/paper_experiments/results/wikitext_run_2026_04_21/RQ3
for m in opt_6.7b gptj_6b bloom_7b1 falcon_7b mistral_7b_v03 llama2_13b; do
  f=$m/exp5_detailed_results.json
  [ -f $f ] || { echo "MISS $f"; continue; }
  python3 - <<PY
import json
d = json.load(open("$f"))
ws = d.get("word_stats", {})
nf = sum(1 for v in ws.values() if v.get("is_function"))
ns = sum(1 for v in ws.values() if v.get("is_structural"))
nc = sum(1 for v in ws.values() if not v.get("is_function") and not v.get("is_structural"))
status = "OK" if (nc > 0 and nf > 0) else "BAD"
print(f"{status}  $m  func={nf} struct={ns} content={nc}")
PY
done
```

**Pass criteria**: all 6 show `OK` with `content > 0` 且 `func > 0`。

### Task 5.3 — RQ6 peak_layer field (B5/B6 validation)

```bash
cd ~/ma/paper_experiments/results/wikitext_run_2026_04_21
for m in opt_6.7b gptj_6b bloom_7b1 falcon_7b mistral_7b_v03 llama2_13b; do
  f="RQ6/$m/v_ablation_results.json"
  if [ -f "$f" ]; then
    has_peak=$(jq -r 'keys[] as $k | select($k | test("peak_layer"))' "$f" 2>/dev/null | head -1)
    echo "OK $m peak_layer_field=${has_peak:+yes}"
  else
    echo "MISS $m"
  fi
done
```

### Task 5.4 — opt_6.7b ANOMALY records

```bash
cat ~/ma/paper_experiments/results/wikitext_run_2026_04_21/RQ2a/opt_6.7b/*.json 2>/dev/null | jq '.' | head -40
cat ~/ma/paper_experiments/results/wikitext_run_2026_04_21/RQ2c/opt_6.7b/*rq6_greedy.json 2>/dev/null | jq '.' | head -40
```

Archive for paper write-up。

---

## Phase 6 — Sync results back to primary (≈5 min)

### Task 6.1 — rsync results

From secondary:

```bash
cd ~/ma/paper_experiments/results
rsync -avP --stats wikitext_run_2026_04_21/ \
    root@117.50.223.194:/root/changeHead_massvieAcitve/paper_experiments/results/wikitext_run_2026_04_21/
```

**Pass criteria**: file count 一致。

### Task 6.2 — Sync pipeline log

```bash
LATEST=$(ls -td ~/ma/paper_experiments/logs/pipeline_* | head -1)
rsync -avP ${LATEST}/ \
    root@117.50.223.194:/root/changeHead_massvieAcitve/paper_experiments/logs/${LATEST##*/}_secondary/
```

### Task 6.3 — Confirm unified count on primary

SSH 到 primary：
```bash
cd /root/changeHead_massvieAcitve/paper_experiments/results/wikitext_run_2026_04_21
for rq in RQ1 RQ2a RQ2c RQ3 RQ4 RQ5s RQ5m RQ6; do
  n=$(ls -d $rq/*/ 2>/dev/null | wc -l)
  echo "$rq: $n models"
done
```

**Pass criteria**: 每个 RQ 大约 21-24 models（取决于 primary 跑到几个）。

---

## Recovery / Troubleshooting

### §R1 — HF access blocked

1. **Mirror from primary**（如果 primary 能下）：
   ```bash
   cd ~/.cache/huggingface/hub
   tar czf /tmp/six_models_hf.tar.gz models--bigscience--bloom-7b1 models--tiiuae--falcon-7b \
       models--EleutherAI--gpt-j-6b models--meta-llama--Llama-2-13b-hf \
       models--mistralai--Mistral-7B-v0.3 models--facebook--opt-6.7b
   ```
   scp 到 secondary 后 untar 到 `$HF_HOME/hub/`。
2. **镜像**：`export HF_ENDPOINT=https://hf-mirror.com`。
3. **放弃 secondary，在 primary 顺序跑**。

### §R2 — OOM

Order of mitigations:
1. `--nsamples 30 → 20`
2. 让 load_model.py 把该模型改 bf16
3. `attn_implementation=sdpa`

### §R3 — transformers 版本不匹配

Option A：升级
```bash
pip install -U 'transformers>=5.0'
```

Option B：回滚 B8
```bash
cp paper_experiments/_backup_*/monkey_patch/modify_llama.py paper_experiments/monkey_patch/modify_llama.py
```

### §R4 — 单个 pair 失败

```bash
python3 paper_experiments/fixes/run_pipeline.py \
    --models <m> --rqs <rq> --nsamples 30 --timeout 7200
```

### §R5 — No L_ORIGIN

```bash
python3 -c "import json; d=json.load(open('paper_experiments/origin_layer/output/L_ORIGIN.json')); \
  [print(m, d.get(m, 'MISSING')) for m in ['bloom_7b1','falcon_7b','gptj_6b','llama2_13b','mistral_7b_v03','opt_6.7b']]"
```

期望：
```
bloom_7b1 3
falcon_7b 3
gptj_6b 2
llama2_13b 0
mistral_7b_v03 0
opt_6.7b 1
```

### §R6 — MLP naming mismatch

已覆盖所有 6 个。若新报错 `ValueError: Cannot identify MLP submodules` → 在 `paper_experiments/fixes/lib/model_utils.py` 加白名单。

---

## Quick reference

```bash
# 进度
tail -20 ~/ma/paper_experiments/logs/secondary_sweep.log
ls -td ~/ma/paper_experiments/logs/pipeline_* | head -1

# 重跑单个
python3 paper_experiments/fixes/run_pipeline.py --models <m> --rqs <rq> --nsamples 30

# 同步回 primary
rsync -avP ~/ma/paper_experiments/results/wikitext_run_2026_04_21/ \
    root@117.50.223.194:/root/changeHead_massvieAcitve/paper_experiments/results/wikitext_run_2026_04_21/
```

---

## Change log

| Date | Change |
|---|---|
| 2026-04-21 | Initial version — 6 HF-download models on secondary server |
