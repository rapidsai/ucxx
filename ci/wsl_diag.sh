#!/bin/bash
# TEMP diagnostics for WSL Python benchmark slowness
set -uo pipefail
b() {
  local backend=$1 pm=$2 aw=$3 ds=$4 pf=$5 nb=$6
  SECONDS=0
  UCX_KEEPALIVE_INTERVAL=1ms UCXPY_ENABLE_DELAYED_SUBMISSION=$ds UCXPY_ENABLE_PYTHON_FUTURE=$pf timeout 200 python -m ucxx.benchmarks.send_recv \
    --backend "$backend" -o cupy --reuse-alloc -n 8MiB --n-buffers "$nb" --progress-mode "$pm" $aw >/tmp/out.txt 2>&1
  rc=$?
  echo "$backend $pm aw=$aw ds=$ds pf=$pf nb=$nb rc=$rc t=${SECONDS}s $(grep -E 'Bandwidth \(average\)' /tmp/out.txt | tr -s ' ') $(grep -m1 -E 'Caught signal|Error' /tmp/out.txt)"
}
matrix() {
  b ucxx-core thread "" 0 0 1
  b ucxx-core thread --asyncio-wait 0 0 1
  for pm in blocking thread; do
    for nb in 1 8; do
      b ucxx-async $pm "" 0 0 $nb
      b ucxx-async $pm "" 0 1 $nb
      if [ $pm != blocking ]; then
        b ucxx-async $pm "" 1 0 $nb
        b ucxx-async $pm "" 1 1 $nb
      fi
    done
  done
}
echo "##### UCX_TLS=^cuda_ipc UCX_TCP_{TX,RX}_SEG_SIZE=1m"
UCX_TLS=^cuda_ipc UCX_TCP_TX_SEG_SIZE=1m UCX_TCP_RX_SEG_SIZE=1m matrix
echo "##### UCX_TLS=all"
UCX_TLS=all matrix
