#!/usr/bin/env bash
set -e

# 1) ожидаем, что RAY_ADDRESS уже есть в окружении
#RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
#RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:9999"}

CKPTS_DIR="${CKPTS_DIR:-"${PWD}/ckpts"}"


KEEP_LAST=2
PERIOD_SEC=15

SENTINEL="$CKPTS_DIR/.cleaner.run"


# robust, nounset-proof cleaner
clean_loop() (
  set +u                    # isolate from global 'set -u' in a subshell
  shopt -s nullglob         # empty globs => empty array, not literal
  local -a cps=()
  local n=0
  local d

  while [[ -f "$SENTINEL" ]]; do
    # 1) Find checkpoints and sort them
    cps=( "$CKPTS_DIR"/global_step_* )
    if ((${#cps[@]})); then
      IFS=$'\n' cps=( $(printf '%s\n' "${cps[@]}" | sort -V) )
    fi
    n=${#cps[@]}

    # 2) Remove everything except the newest KEEP_LAST
    if (( n > KEEP_LAST )); then
      for d in "${cps[@]:0:n-KEEP_LAST}"; do
        rm -rf -- "$d"
      done
    fi

    # 3) Re-scan survivors and strip huggingface dirs in all but newest
    cps=( "$CKPTS_DIR"/global_step_* )
    if ((${#cps[@]})); then
      IFS=$'\n' cps=( $(printf '%s\n' "${cps[@]}" | sort -V) )
    fi
    n=${#cps[@]}
    if (( n > 1 )); then
      for d in "${cps[@]:0:n-1}"; do
        rm -rf -- "$d/actor/huggingface"
      done
    fi

    sleep "$PERIOD_SEC"
  done
)

: > "$SENTINEL"                 # 1) create sentinel
clean_loop & CLEAN_PID=$!    # 2) launch cleaner in background
#trap 'rm -f "$SENTINEL"; kill "$CLEAN_PID" 2>/dev/null || true' EXIT INT TERM

JOB_ID="train-$(date +%s)-$RANDOM"

# 2) сабмитим
teray job submit --no-wait --submission-id "$JOB_ID" \
  --working-dir . \
  -- python job.py

# blocks until the job finishes
teray job logs -f "$JOB_ID" || true

rm -f "$SENTINEL"               # cleaner stops on next tick

kill "$CLEAN_PID" 2>/dev/null || true
wait "$CLEAN_PID" 2>/dev/null || true
trap - EXIT INT TERM
