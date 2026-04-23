# Primary Server Remaining Debug Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete remaining debug experiments (RQ5 V-ablation rerun, RQ6 exp6 rerun, RQ1 qwen2_7b Inf fix) on **primary server 117.50.223.194** for 14 dense models with weights available locally, then rsync results back to local Mac and merge into论点 E 总表.

**Architecture:** Use `/usr/local/miniconda3/envs/py312/bin/python` on primary. Run each experiment script with `--layer_id = L_origin` per model. Save all outputs under `/tmp/systemd_<exp>/` to avoid the (now-killed) crypto-miner's process-killer pattern that scans for command lines not containing "systemd". Use GPU gate thresholds to avoid OOM with other potential users. Tar + ssh pipe results back to `fixes/results_stage2/`.

**Tech Stack:**
- Primary server: 117.50.223.194:23 root/j053v429E1a8LNQs
- Python: `/usr/local/miniconda3/envs/py312/bin/python`
- Repo: `/root/changeHead_massvieAcitve/paper_experiments/`
- Scripts: `RQ5_v_matrix_ablation/exp5_v_ablation.py`, `RQ6_single_layer_activation/exp6_single_layer_activation.py`, `RQ1_attention_contribution/exp1_feasibility_test.py`
- Local Mac output: `/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/`

---

## File Structure

### Scripts to create

- Create: `/tmp/systemd_agentF_rq5.sh` (remote) — RQ5 起源层 batch runner with GPU gate
- Create: `/tmp/systemd_agentG_rq6exp6.sh` (remote) — RQ6 exp6 batch runner
- Create: `/tmp/systemd_agentH_rq1patch.sh` (remote) — RQ1 qwen2_7b nsamples=60 single run

### Scripts to reuse (unchanged)

- `RQ5_v_matrix_ablation/exp5_v_ablation.py` — per-model V-ablation with random orthogonal replacement at L_origin
- `RQ6_single_layer_activation/exp6_single_layer_activation.py` — per-model top-K keep/remove at L_origin (already has B4/B5/B6 fixes deployed per `fixes/lib/model_utils.py`)
- `RQ1_attention_contribution/exp1_feasibility_test.py` — attention ablation

### Local Mac scripts to reuse

- `paper_experiments/fixes/analyze_HC.py` — already merges 20-model H(C) data
- `paper_experiments/origin_layer/output/L_ORIGIN.json` — single-layer origin layer mapping

### 14 dense models with L_origin (primary has all weights)

| Model | L_origin |
|---|:-:|
| qwen3_0.6b | 2 |
| qwen3_1.7b | 0 |
| qwen3_4b | 0 |
| qwen3_8b | 2 |
| qwen3_14b | 6 |
| qwen3_32b | 6 |
| qwen2_7b | 3 |
| qwen2.5_7b | 3 |
| qwen1.5_14b | 3 |
| qwen3.5_9b | 22 |
| qwen3.5_27b | 9 |
| glm4_9b | 1 |
| llama3.1_8b | 1 |
| yi_9b | 8 |

---

## Task 1: Verify primary state clean

**Files:**
- Check only, no modifications

- [ ] **Step 1: Verify miner not resurrected and code state clean**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "echo '=== miner ==='; ls /root/.sys-cache 2>&1 || echo NO_MINER_DIR; pgrep -af 'sys-cache|free_proc|xmrig' | grep -v grep || echo NO_MINER_PROC; echo '=== GPU ==='; nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; echo '=== code fixes check ==='; grep -c 'B8 fix\\|B13\\|B14\\|B15\\|B16' /root/changeHead_massvieAcitve/paper_experiments/lib/model_utils.py /root/changeHead_massvieAcitve/paper_experiments/monkey_patch/modify_llama.py /root/changeHead_massvieAcitve/paper_experiments/lib/load_model.py"
```

Expected:
- `NO_MINER_DIR` and `NO_MINER_PROC` — miner stays dead
- GPU free ≥ 60 GB, used ≤ 20 GB (by other users' misc jobs, OK)
- Each code file shows non-zero count of bug-fix markers

- [ ] **Step 2: If miner resurrected, abort and alert user**

If `/root/.sys-cache/` exists or processes named `sys-cache` / `free_proc` running:
```bash
echo "ALERT: crypto-miner resurrected on primary server. Stopping plan execution."
```
Do not proceed. Escalate to user for `docker restart` + password rotation.

- [ ] **Step 3: Commit nothing**

This is a state verification task only. No commit needed.

---

## Task 2: Pilot RQ5 V-ablation on qwen3_0.6b

Validate RQ5 script works at L_origin=2 before batch. Expected: `exp5_v_ablation.py` produces `ΔMA ≤ -80%` for a small model at origin layer per论点 E prediction.

**Files:**
- Local Mac: check existing `fixes/results_stage2/RQ5_pilot/qwen3_0.6b/` for pre-existing data (avoid re-run if complete)
- Remote: `/root/changeHead_massvieAcitve/paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation.py` (no modifications)

- [ ] **Step 1: Check if RQ5 script exists and has known-good args**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls /root/changeHead_massvieAcitve/paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation.py; grep -n 'argparse\\|add_argument' /root/changeHead_massvieAcitve/paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation.py | head -15"
```

Expected args at minimum: `--model`, `--layer_id`, `--nsamples`, `--savedir`.

- [ ] **Step 2: Run pilot on qwen3_0.6b L=2, nsamples=10 (fast)**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cd /root/changeHead_massvieAcitve/paper_experiments && mkdir -p /tmp/systemd_rq5/qwen3_0.6b && /usr/local/miniconda3/envs/py312/bin/python -u RQ5_v_matrix_ablation/exp5_v_ablation.py --model qwen3_0.6b --layer_id 2 --nsamples 10 --savedir /tmp/systemd_rq5/qwen3_0.6b 2>&1 | tail -30"
```

Expected: script finishes successfully and prints a `ΔMA` or `MA_ratio` value. If crash, abort Task 2 and investigate in Task 2 retry.

- [ ] **Step 3: Inspect pilot output**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls -la /tmp/systemd_rq5/qwen3_0.6b/; cat /tmp/systemd_rq5/qwen3_0.6b/*.json 2>/dev/null | head -40"
```

Expected: at least one `.json` file with fields like `baseline_ma_top1` and `ablated_ma_top1` (or equivalent), with `ablated < baseline`.

- [ ] **Step 4: Commit nothing (remote-only outputs)**

No git commit. Outputs stay on remote `/tmp/systemd_rq5/`.

---

## Task 3: Write the RQ5 batch runner

Create a serial batch runner for 14 dense models at L_origin with GPU gate.

**Files:**
- Create (remote): `/tmp/systemd_agentF_rq5.sh`

- [ ] **Step 1: Write the batch script via heredoc over ssh**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cat > /tmp/systemd_agentF_rq5.sh" <<'EOF'
#!/bin/bash
cd /root/changeHead_massvieAcitve/paper_experiments
mkdir -p /tmp/systemd_rq5/_logs
declare -A LAYERS=(
  [qwen3_0.6b]=2 [qwen3_1.7b]=0 [qwen3_4b]=0 [qwen3_8b]=2
  [qwen3_14b]=6 [qwen3_32b]=6 [qwen2_7b]=3 [qwen2.5_7b]=3
  [qwen1.5_14b]=3 [qwen3.5_9b]=22 [qwen3.5_27b]=9
  [glm4_9b]=1 [llama3.1_8b]=1 [yi_9b]=8
)
MODELS="qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b llama3.1_8b glm4_9b yi_9b qwen2_7b qwen2.5_7b qwen3.5_9b qwen1.5_14b qwen3_14b qwen3.5_27b qwen3_32b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in $MODELS; do
  L=${LAYERS[$M]}
  echo "[$(date)] === $M L=$L ==="
  # GPU gate: wait until ≥ 20 GB free
  while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 20000 ]; then break; fi
    echo "[$(date)] gate: free=${FREE}MiB < 20000, sleep 60"
    sleep 60
  done
  SAVE=/tmp/systemd_rq5/$M
  mkdir -p $SAVE
  /usr/local/miniconda3/envs/py312/bin/python -u RQ5_v_matrix_ablation/exp5_v_ablation.py \
    --model $M --layer_id $L --nsamples 30 --savedir $SAVE \
    > /tmp/systemd_rq5/_logs/${M}.log 2>&1
  RC=$?
  echo "[$(date)] done $M rc=$RC"
done
echo "[$(date)] ALL_RQ5_DONE"
EOF
```

- [ ] **Step 2: Verify script written**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "head -5 /tmp/systemd_agentF_rq5.sh; wc -l /tmp/systemd_agentF_rq5.sh"
```

Expected: file has ~25+ lines, starts with `#!/bin/bash`.

- [ ] **Step 3: Commit nothing (remote script)**

Script lives on remote only.

---

## Task 4: Execute RQ5 batch for 14 models

**Files:**
- Execute (remote): `/tmp/systemd_agentF_rq5.sh`
- Output (remote): `/tmp/systemd_rq5/<model>/`

- [ ] **Step 1: Launch batch in background**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "setsid nohup bash /tmp/systemd_agentF_rq5.sh > /tmp/systemd_rq5/_runall.log 2>&1 </dev/null & disown
sleep 3
pgrep -af 'systemd_agentF_rq5\\|exp5_v_ablation' | grep -v grep | head"
```

Expected: at least one process shown running `/tmp/systemd_agentF_rq5.sh` or `exp5_v_ablation.py`.

- [ ] **Step 2: Poll progress every 5 min until done**

Run (repeat until `ALL_RQ5_DONE` appears in runall log or all 14 JSONs exist):
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls /tmp/systemd_rq5/*/exp5*.json 2>/dev/null | wc -l; tail -5 /tmp/systemd_rq5/_runall.log"
```

Expected ETA: ~25 min (14 models × ~1.5 min each including GPU gate).

- [ ] **Step 3: Verify all 14 have JSON output**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls /tmp/systemd_rq5/; ls /tmp/systemd_rq5/*/exp5*.json 2>/dev/null | wc -l"
```

Expected: `14` JSON files.

If fewer than 14, check `_logs/<model>.log` for each missing model to diagnose.

- [ ] **Step 4: tar + pipe results back to local Mac**

Run (from local Mac):
```bash
mkdir -p /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ5_stage2
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cd /tmp && tar czf - systemd_rq5" | tar -xz -C /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/ --transform 's|^systemd_rq5|RQ5_stage2|'
ls /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ5_stage2/
```

Expected: 14 model subdirs visible locally.

- [ ] **Step 5: Commit results to local git**

```bash
cd /Users/a1-6/importantfile/Research/ma
git add paper_experiments/fixes/results_stage2/RQ5_stage2/
git commit -m "feat(RQ5): stage 2 V-ablation results for 14 dense models at L_origin"
```

---

## Task 5: Write the RQ6 exp6 batch runner

Rerun `exp6_single_layer_activation.py` (top-K / remove / keep) after B4/B5/B6 bug fixes. The bug was: `get_critical_layer()` defaulted to L0 and baseline was measured at critical_layer instead of peak MA layer.

**Files:**
- Create (remote): `/tmp/systemd_agentG_rq6exp6.sh`
- Reuse (remote): `/root/changeHead_massvieAcitve/paper_experiments/RQ6_single_layer_activation/exp6_single_layer_activation.py`

- [ ] **Step 1: Verify RQ6 script has B4/B5/B6 fixes applied on primary**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "grep -cE 'B4|B5|B6|L_ORIGIN' /root/changeHead_massvieAcitve/paper_experiments/RQ6_single_layer_activation/exp6_single_layer_activation.py /root/changeHead_massvieAcitve/paper_experiments/lib/model_utils.py"
```

Expected: non-zero counts indicating fixes are in place.

- [ ] **Step 2: If fixes missing, scp local `fixes/` version over**

Run (local Mac):
```bash
# Compare local vs remote
diff <(ssh -p 23 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@117.50.223.194 "cat /root/changeHead_massvieAcitve/paper_experiments/RQ6_single_layer_activation/exp6_single_layer_activation.py") /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/RQ6_v_ablation/exp6_single_layer_activation.py 2>&1 | head -30
# If diff exists and local has newer B4/B5/B6 fixes, push:
sshpass -p 'j053v429E1a8LNQs' scp -P 23 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/RQ6_v_ablation/exp6_single_layer_activation.py root@117.50.223.194:/root/changeHead_massvieAcitve/paper_experiments/RQ6_single_layer_activation/exp6_single_layer_activation.py
```

Expected: Either diff is empty (no action) or scp succeeds.

- [ ] **Step 3: Write the batch script**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cat > /tmp/systemd_agentG_rq6exp6.sh" <<'EOF'
#!/bin/bash
cd /root/changeHead_massvieAcitve/paper_experiments
mkdir -p /tmp/systemd_rq6exp6/_logs
declare -A LAYERS=(
  [qwen3_0.6b]=2 [qwen3_1.7b]=0 [qwen3_4b]=0 [qwen3_8b]=2
  [qwen3_14b]=6 [qwen3_32b]=6 [qwen2_7b]=3 [qwen2.5_7b]=3
  [qwen1.5_14b]=3 [qwen3.5_9b]=22 [qwen3.5_27b]=9
  [glm4_9b]=1 [llama3.1_8b]=1 [yi_9b]=8
)
MODELS="qwen3_0.6b qwen3_1.7b qwen3_4b qwen3_8b llama3.1_8b glm4_9b yi_9b qwen2_7b qwen2.5_7b qwen3.5_9b qwen1.5_14b qwen3_14b qwen3.5_27b qwen3_32b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in $MODELS; do
  L=${LAYERS[$M]}
  echo "[$(date)] === $M L=$L ==="
  while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE" -ge 20000 ]; then break; fi
    echo "[$(date)] gate: free=${FREE}MiB < 20000, sleep 60"
    sleep 60
  done
  SAVE=/tmp/systemd_rq6exp6/$M
  mkdir -p $SAVE
  /usr/local/miniconda3/envs/py312/bin/python -u RQ6_single_layer_activation/exp6_single_layer_activation.py \
    --model $M --layer_id $L --nsamples 30 --savedir $SAVE \
    > /tmp/systemd_rq6exp6/_logs/${M}.log 2>&1
  RC=$?
  echo "[$(date)] done $M rc=$RC"
done
echo "[$(date)] ALL_RQ6EXP6_DONE"
EOF
```

- [ ] **Step 4: Commit nothing (remote script)**

Script lives on remote.

---

## Task 6: Execute RQ6 exp6 batch for 14 models

**Files:**
- Execute (remote): `/tmp/systemd_agentG_rq6exp6.sh`
- Output (remote): `/tmp/systemd_rq6exp6/<model>/`

- [ ] **Step 1: Launch in background**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "setsid nohup bash /tmp/systemd_agentG_rq6exp6.sh > /tmp/systemd_rq6exp6/_runall.log 2>&1 </dev/null & disown
sleep 3
pgrep -af 'systemd_agentG_rq6exp6\\|exp6_single_layer' | grep -v grep | head"
```

Expected: at least one process running.

- [ ] **Step 2: Poll progress every 10 min until done**

Run (repeat until done):
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls /tmp/systemd_rq6exp6/*/exp6*.json 2>/dev/null | wc -l; tail -5 /tmp/systemd_rq6exp6/_runall.log"
```

Expected ETA: ~60-90 min (14 models × ~4-6 min each; top-K/remove is more expensive than RQ5).

- [ ] **Step 3: Verify all 14 have output**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "ls /tmp/systemd_rq6exp6/; ls /tmp/systemd_rq6exp6/*/exp6*.json 2>/dev/null | wc -l"
```

Expected: `14`.

- [ ] **Step 4: tar + pipe back to local Mac**

Run (local Mac):
```bash
mkdir -p /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ6_stage2
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cd /tmp && tar czf - systemd_rq6exp6" | tar -xz -C /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/ --transform 's|^systemd_rq6exp6|RQ6_stage2|'
ls /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ6_stage2/
```

Expected: 14 model subdirs.

- [ ] **Step 5: Commit**

```bash
cd /Users/a1-6/importantfile/Research/ma
git add paper_experiments/fixes/results_stage2/RQ6_stage2/
git commit -m "feat(RQ6): stage 2 exp6 top-K/remove results for 14 dense models at L_origin"
```

---

## Task 7: qwen2_7b RQ1 with nsamples=60 (Inf data fix)

RQ1 qwen2_7b baseline was ≈ 0 causing `ΔTop1 = +∞`. Fix per CLAUDE.md §3.1: rerun with `nsamples=60` (doubling sample count usually shifts baseline above 0).

**Files:**
- Execute (remote): `/root/changeHead_massvieAcitve/paper_experiments/RQ1_attention_contribution/exp1_feasibility_test.py`
- Output (remote): `/tmp/systemd_rq1_qwen2_7b/`

- [ ] **Step 1: Run qwen2_7b RQ1 with nsamples=60**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "mkdir -p /tmp/systemd_rq1_qwen2_7b && cd /root/changeHead_massvieAcitve/paper_experiments && /usr/local/miniconda3/envs/py312/bin/python -u RQ1_attention_contribution/exp1_feasibility_test.py --model qwen2_7b --nsamples 60 --savedir /tmp/systemd_rq1_qwen2_7b 2>&1 | tail -40"
```

Expected: completes with `baseline_top1 > 0` and `disabled_top1 / baseline_top1` ratio finite.

- [ ] **Step 2: Verify output has finite ratio**

Run:
```bash
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cat /tmp/systemd_rq1_qwen2_7b/*.json 2>/dev/null | grep -E 'baseline|disabled|residual|delta_ma' | head -10"
```

Expected: numeric values, not `Infinity` or `NaN`.

- [ ] **Step 3: tar back**

```bash
mkdir -p /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ1_patches
sshpass -p 'j053v429E1a8LNQs' ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23 root@117.50.223.194 "cd /tmp && tar czf - systemd_rq1_qwen2_7b" | tar -xz -C /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ1_patches/
ls /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ1_patches/
```

Expected: `systemd_rq1_qwen2_7b/` subdir visible.

- [ ] **Step 4: Commit**

```bash
cd /Users/a1-6/importantfile/Research/ma
git add paper_experiments/fixes/results_stage2/RQ1_patches/
git commit -m "fix(RQ1): qwen2_7b rerun nsamples=60 to resolve baseline≈0 Infinity"
```

---

## Task 8: Update EXPERIMENT_PLAN.md with new RQ5/RQ6 数据总结

**Files:**
- Modify: `paper_experiments/docs/EXPERIMENT_PLAN.md` — append new RQ5 / RQ6 sections with 14-model data

- [ ] **Step 1: Aggregate RQ5 Δ MA across 14 models into a summary**

Run (local Mac):
```bash
python3 <<'PYEOF'
import json, os, glob
ROOT = '/Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2/RQ5_stage2'
print(f"{'Model':<20} {'ΔMA%':>8}")
print('-' * 35)
for d in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, d)
    if not os.path.isdir(path): continue
    js = glob.glob(os.path.join(path, '*.json'))
    if not js: continue
    data = json.load(open(js[0]))
    baseline = data.get('baseline_ma_top1') or data.get('baseline_top1') or 0
    ablated = data.get('ablated_ma_top1') or data.get('ablated_top1') or 0
    if baseline > 0:
        delta = (ablated - baseline) / baseline * 100
        print(f"{d:<20} {delta:>7.1f}%")
PYEOF
```

Expected: table showing per-model ΔMA, most with < -80% if论点 E holds at origin layer.

- [ ] **Step 2: Append RQ5 results section to EXPERIMENT_PLAN.md**

Use Edit tool to insert after the existing RQ5 section, with the actual numbers from Step 1 rendered into a markdown table. Do not use placeholders.

- [ ] **Step 3: Repeat for RQ6 exp6 top-K/remove summary**

Run similar aggregation on `fixes/results_stage2/RQ6_stage2/*/` — extract `remove_top_k_ma_ratio` or `keep_top_k_ma_ratio` per model.

- [ ] **Step 4: Commit**

```bash
cd /Users/a1-6/importantfile/Research/ma
git add paper_experiments/docs/EXPERIMENT_PLAN.md
git commit -m "docs(EXPERIMENT_PLAN): add RQ5 + RQ6 stage 2 results tables (14 dense models)"
```

---

## Self-Review (Post-writing)

### Spec coverage check

| Requirement | Task | Status |
|---|---|---|
| RQ5 起源层 rerun (论点 E 因果验证) | Task 2-4 | ✓ covered |
| RQ6 exp6 rerun (B4/B5/B6 修完 baseline 错层) | Task 5-6 | ✓ covered |
| qwen2_7b RQ1 Inf 修复 | Task 7 | ✓ covered |
| Miner 复活检查 | Task 1 | ✓ covered |
| 数据回 local Mac + git commit | Task 4/6/7 Step 4/5 | ✓ covered |
| EXPERIMENT_PLAN.md 更新 | Task 8 | ✓ covered |
| GPU gate 避免 OOM | Task 3/5 | ✓ covered |

### Placeholder scan

No "TBD", no "TODO implement later", no "Similar to Task N", no unreferenced function names. All commands complete.

### Type consistency

- Every task uses `L_origin` from the same 14-model table (Task 3 Step 1 and Task 5 Step 3 identical `LAYERS` assoc array).
- Python interpreter path constant: `/usr/local/miniconda3/envs/py312/bin/python` everywhere.
- SSH options string constant: `-o ConnectTimeout=20 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 23`.
- Savedir prefix constant: `/tmp/systemd_<exp>/<model>/` on remote.

### Scope check

Out of scope (reasons documented):
- gpt2 / qwen2.5_0.5b / llama2_7b_chat — 权重不在 primary（user confirmed 这些在 secondary 8.138.30.52）
- glm4_32b — 权重不在 primary + fp16 Inf 需 fp32 分支
- qwen3_30b_a3b / qwen3.5_35b_a3b — MoE Tier C 延办
- u₁ 抽取脚本修 (gptj/opt) — 那俩模型在 secondary 跑过了，extraction bug 和 primary 运行无关

These should be scheduled as a **separate plan for secondary server** or Tier C / delay.

---

## Execution Handoff

Plan complete and saved to `paper_experiments/docs/plans/2026-04-22-primary-remaining-debug.md`.

Estimated runtime: Task 1–7 ≈ **1.5–2 hours** total (RQ6 exp6 is longest at 60–90 min).

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each task is self-contained.

**2. Inline Execution** — Execute tasks in this session, batch with checkpoints.

Which approach?
