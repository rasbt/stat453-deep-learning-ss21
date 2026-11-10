#!/usr/bin/env bash
# Redo the DDIM half of run_eval.sh (after a sampler change); keeps existing DDPM FID rows.
#   scripts/reeval_ddim.sh <run> <device> [extra args, e.g. --no-ema]
set -u
RUN=$1; DEV=$2; shift 2; EXTRA="$*"
PY=${PY:-python}
CK=checkpoints/$RUN/ddpm_checkpoint.pt
CSV=results/$RUN/fid.csv
echo "=== reeval ddim: $RUN on $DEV $EXTRA ($(date))"
if [ -f "$CSV" ]; then awk -F, 'NR==1 || $7=="ddpm"' "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"; fi
$PY scripts/sample.py        --checkpoint $CK --seed 0 --device $DEV --num-samples 100 --sampler ddim --steps 50 $EXTRA
$PY scripts/compare_steps.py --checkpoint $CK --seed 0 --device $DEV $EXTRA
for S in 10 20 50 100; do
  $PY scripts/evaluate.py --checkpoint $CK --sampler ddim --steps $S -n 5000 --device $DEV --label $RUN $EXTRA
done
echo "=== done $RUN ($(date))"; cat $CSV
