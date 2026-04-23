#!/bin/bash
# Run on LOCAL Mac to pull results back when master runner finishes
mkdir -p /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2_primary_missing
sshpass -p 'j053v429E1a8LNQs' ssh -p 23 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@117.50.223.194 \
  "cd /tmp && tar czf - systemd_rq1_miss systemd_rq2a_miss systemd_rq3_miss systemd_rq4_miss systemd_rq5_miss systemd_rq6exp6_miss systemd_hc_miss systemd_u1_miss.json systemd_logs_agentI 2>/dev/null" | \
  tar -xzv -C /Users/a1-6/importantfile/Research/ma/paper_experiments/fixes/results_stage2_primary_missing/
