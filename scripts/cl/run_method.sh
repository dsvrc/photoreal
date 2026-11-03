#!/bin/bash
# One continual-learning method end to end on the body model, sessions PXB184 -> RLW104 -> TXB805 -> GQS883.
# For each session k = 1..3: build the buffer of session k-1 with the model after k-1, train session k
# (warm start, fresh AdamW) replaying the buffers of sessions 0..k-1, and test it on every subject seen so far.
# Every step is skipped when its output exists, so running it again continues where it stopped.
#
#   bash scripts/cl/run_method.sh <vanilla|er|star|ocs|gcr|mir|csrel|tmr|tmr_joint>
#
# Optional environment: STEPS (60000), SAVE_EVERY (10000), RUNS (runs), BUFFERS (cl_data/buffers),
# EVAL_MODES ("audio_only gt_guide"). Run from the repo root in a2p_env, or submit scripts/cl/run_method.pbs.
set -euo pipefail

METHOD=${1:?usage: run_method.sh <vanilla|er|star|ocs|gcr|mir|csrel|tmr|tmr_joint>}
STEPS=${STEPS:-60000}
SAVE_EVERY=${SAVE_EVERY:-10000}
RUNS=${RUNS:-runs}
BUFFERS=${BUFFERS:-cl_data/buffers}
EVAL_MODES=${EVAL_MODES:-audio_only gt_guide}
ORDER=(PXB184 RLW104 TXB805 GQS883)
S0=checkpoints/diffusion/c1_pose/model000340000.pt # session 0: the released PXB184 model, shared by all methods

case $METHOD in
  vanilla) SELECT= ;;
  er | star) SELECT=random ;; # Random* (STAR) replays the random buffer
  ocs | gcr | mir | csrel | tmr | tmr_joint) SELECT=$METHOD ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac
TRAIN_EXTRA=()
if [ "$METHOD" = star ]; then TRAIN_EXTRA=(--star); fi

after() { # checkpoint at the end of session $1
  if [ "$1" -eq 0 ]; then
    echo "$S0"
  else
    printf "%s/%s/s%d_%s/model%09d.pt\n" "$RUNS" "$METHOD" "$1" "${ORDER[$1]}" "$STEPS"
  fi
}

python -m cl.evaluate --ckpt "$S0" --subjects PXB184 --modes $EVAL_MODES

REPLAY=()
for k in 1 2 3; do
  j=$((k - 1))
  if [ -n "$SELECT" ]; then
    buf=$BUFFERS/$METHOD/s${j}_${ORDER[$j]}.json
    if [ ! -f "$buf" ]; then
      PREV=()
      if [ "$METHOD" = csrel ] && [ "$j" -gt 0 ]; then PREV=(--prev_ckpt "$(after $((j - 1)))"); fi
      echo "== $METHOD: buffer of session $j (${ORDER[$j]})"
      python -m cl.build_buffer --method "$SELECT" --subject "${ORDER[$j]}" --session "$j" \
        --ckpt "$(after "$j")" --out "$buf" ${PREV[@]+"${PREV[@]}"}
    fi
    REPLAY+=("$buf")
  fi

  final=$(after "$k")
  if [ ! -f "$final" ]; then
    echo "== $METHOD: session $k (${ORDER[$k]})"
    python -m cl.train_session --init_ckpt "$(after "$j")" --data_root "dataset/${ORDER[$k]}" \
      --save_dir "$(dirname "$final")" --steps "$STEPS" --save_every "$SAVE_EVERY" --resume \
      ${REPLAY[@]+--replay "${REPLAY[@]}"} ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}
  fi
  echo "== $METHOD: test after session $k"
  python -m cl.evaluate --ckpt "$final" --subjects "${ORDER[@]:0:$((k + 1))}" --modes $EVAL_MODES
done

python -m cl.summarize --methods "$METHOD" --runs "$RUNS" --steps "$STEPS" --modes $EVAL_MODES
