#!/bin/bash
# Pass-2 full-auto pipeline:
#   wait for distill pass 2 (win_mac + ubuntu) -> dedup -> join to SpecForge
#   format -> shard -> symlink-farm guard -> shard-chain incremental training
#   warm-started from the pass-1 final ckpt (part-07 step2912).
#
# Usage: nohup bash pass2_auto.sh > pass2_auto.log 2>&1 &
set -uo pipefail

BASE=/mnt/lisiqi23/internship/zhangyikang/uiagent/pc_agent_distill
D=$BASE/dflash2
PASS1_FINAL_CKPT=$D/train-part-07/guiowl32b-dflash2-part-07-step2912

log() { echo "[pass2_auto $(date '+%m-%d %H:%M:%S')] $*"; }

wait_distill_done() {
    # $1 = status json path, $2 = label
    while true; do
        if [ -f "$1" ]; then
            DONE=$(python3 -c "import json;s=json.load(open('$1'));print(s['done'])" 2>/dev/null || echo 0)
            TOTAL=$(python3 -c "import json;s=json.load(open('$1'));print(s['total'])" 2>/dev/null || echo -1)
            if [ "$DONE" = "$TOTAL" ] && [ "$DONE" -gt 0 ] 2>/dev/null; then
                log "$2 distillation done: $DONE/$TOTAL"
                return 0
            fi
            log "$2 distillation: $DONE/$TOTAL"
        else
            log "$2 status file not yet: $1"
        fi
        sleep 300
    done
}

dedup() {
    # $1 raw, $2 deduped
    python3 - "$1" "$2" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
seen = set()
n = 0
with open(src) as fin, open(dst, 'w') as fout:
    for line in fin:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if 'error' in r or r['sample_id'] in seen:
            continue
        seen.add(r['sample_id'])
        fout.write(line)
        n += 1
print(f"dedup {src} -> {dst}: {n} unique", flush=True)
PYEOF
}

log "waiting for win_mac pass-2 distillation..."
wait_distill_done $BASE/distill_32b_pass2_status.json win_mac
log "waiting for ubuntu pass-2 distillation..."
wait_distill_done $BASE/distill_ubuntu_32b_pass2_status.json ubuntu

log "dedup pass-2 outputs"
dedup $BASE/distill_32b_pass2_raw.jsonl $BASE/distill_32b_pass2_dedup.jsonl || exit 1
dedup $BASE/distill_ubuntu_32b_pass2_raw.jsonl $BASE/distill_ubuntu_32b_pass2_dedup.jsonl || exit 1

log "join to SpecForge format"
python3 $D/join_convert.py --input $BASE/agentnet_ubuntu_5k_pcagent.jsonl \
    --distill $BASE/distill_ubuntu_32b_pass2_dedup.jsonl \
    --image-root /dev/shm/pc_agent_images/ubuntu \
    --output $D/data_pass2_ubuntu.jsonl || exit 1
python3 $D/join_convert.py --input $BASE/agentnet_win_mac_pcagent.jsonl \
    --distill $BASE/distill_32b_pass2_dedup.jsonl \
    --image-root /dev/shm/pc_agent_images/win_mac \
    --output $D/data_pass2_win_mac.jsonl || exit 1
# join_convert dedups output by id inline? No: dedup here like pass 1
python3 - "$D" <<'PYEOF'
import json, sys, os
D = sys.argv[1]
for name in ['data_pass2_ubuntu', 'data_pass2_win_mac']:
    src = os.path.join(D, name + '.jsonl')
    seen = set()
    out = []
    for line in open(src):
        rid = json.loads(line)['id']
        if rid in seen:
            continue
        seen.add(rid)
        out.append(line)
    with open(src, 'w') as f:
        f.writelines(out)
    print(f"{name}: {len(out)} after dedup", flush=True)
PYEOF
cat $D/data_pass2_ubuntu.jsonl $D/data_pass2_win_mac.jsonl > $D/data_pass2_all.jsonl
log "pass-2 data assembled: $(wc -l < $D/data_pass2_all.jsonl) samples"

log "sharding into parts_pass2/"
mkdir -p $D/parts_pass2
split -n l/8 -d --additional-suffix=.jsonl $D/data_pass2_all.jsonl $D/parts_pass2/part-
wc -l $D/parts_pass2/*.jsonl | tail -1

log "symlink farm guard"
if ! ls /dev/shm/pc_agent_images/ubuntu/ >/dev/null 2>&1 || \
   [ -z "$(ls /dev/shm/pc_agent_images/ubuntu/ 2>/dev/null | head -1)" ]; then
    python3 $D/make_symlink_farm.py || exit 3
fi

log "start incremental shard-chain training from $PASS1_FINAL_CKPT"
PARTS_DIR=parts_pass2 TAG=pass2- bash $D/run_shard_chain.sh \
    "$PASS1_FINAL_CKPT" \
    part-00 part-01 part-02 part-03 part-04 part-05 part-06 part-07

log "ALL PIPELINE DONE"
