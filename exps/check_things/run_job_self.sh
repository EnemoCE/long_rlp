#!/usr/bin/env bash
set -e

: "${RAY_IP:=localhost}"
export RAY_ADDRESS=${RAY_ADDRESS:-"http://${RAY_IP}:8265"}

teray job submit \
  --address="$RAY_ADDRESS" \
  --working-dir . \
  -- python job.py
