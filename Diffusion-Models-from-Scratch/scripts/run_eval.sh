#!/usr/bin/env bash
# Full evaluation of one run: samples (DDPM + DDIM), step comparison, FID at several step counts.
#   scripts/run_eval.sh <run> <device> [extra args passed to every script, e.g. --no-ema]
set -u
RUN=$1; DEV=$2; shift 2; EXTRA="$*"
PY=${PY:-python}
CK=checkpoints/$RUN/ddpm_checkpoint.pt
echo "=== $RUN on $DEV $EXTRA  ($(date))"
$PY scripts/sample.py        --checkpoint $CK --seed 0 --device $DEV --num-samples 100 $EXTRA
$PY scripts/sample.py        --checkpoint $CK --seed 0 --device $DEV --num-samples 100 --sampler ddim --steps 50 $EXTRA
$PY scripts/compare_steps.py --checkpoint $CK --seed 0 --device $DEV $EXTRA
for S in 10 20 50 100; do
  $PY scripts/evaluate.py --checkpoint $CK --sampler ddim --steps $S -n 5000 --device $DEV --label $RUN $EXTRA
done
$PY scripts/evaluate.py --checkpoint $CK --sampler ddpm -n 5000 --device $DEV --label $RUN $EXTRA
echo "=== done $RUN ($(date))"
