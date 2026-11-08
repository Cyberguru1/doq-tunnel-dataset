#!/usr/bin/env bash
# Generate labeled malicious DoQ tunneling traffic for the dataset.
#
# dnstt is tunneled over DoQ via dnsproxy, entirely on loopback and offline.
# To produce many ML samples from one run, PARALLEL independent tunnel chains
# run concurrently, each with its own dnsproxy-A and dnstt-client on distinct
# ports but sharing one DoQ server (dnsproxy-B) and one dnstt-server:
#
#   workload_i -> dnstt-client_i (UDP 5300+i) -> dnsproxy-A_i (plain DNS -> DoQ)
#              -> dnsproxy-B (DoQ :853 -> plain UDP :53) -> dnstt-server
#              -> local HTTP sink :8000 (files of assorted sizes)
#
# Each dnsproxy-A_i keeps its own QUIC connection to :853, so the capture on
# UDP 853 contains PARALLEL distinct DoQ flows. After 10 s windowing one run
# yields roughly PARALLEL * (seconds / 10) labeled windows. Each chain runs a
# randomized workload (file size + think time) so the flows are not duplicates.
#
# The DoQ transport is handled by dnsproxy on both ends (well tested); dnstt's
# own -doq client was unreliable here ("bad certificate" / "bad rdata").
#
# Usage: ./generate_malicious.sh <output.pcap> [config_label] [seconds]
# Env:   BIN_DIR   dir with dnsproxy, dnstt-server, dnstt-client (default ./bin)
#        PARALLEL  number of concurrent tunnel chains (default 12)
set -euo pipefail

OUT="${1:?usage: generate_malicious.sh <output.pcap> [config_label] [seconds]}"
LABEL="${2:-fast}"
SECONDS_RUN="${3:-120}"
BIN_DIR="${BIN_DIR:-./bin}"
PARALLEL="${PARALLEL:-12}"

DNSTT_SERVER="$BIN_DIR/dnstt-server"
DNSTT_CLIENT="$BIN_DIR/dnstt-client"
DNSPROXY="$BIN_DIR/dnsproxy"

ZONE="t.example.com"
SINK_PORT=8000
TUNNEL_SERVER_PORT=53
CLIENT_DNS_BASE=5300      # dnsproxy-A_i listens on CLIENT_DNS_BASE + i
TUNNEL_LOCAL_BASE=7000    # dnstt-client_i listens on TUNNEL_LOCAL_BASE + i
RESOLVER_PORT=853         # shared DoQ port (dnsproxy-A_i -> dnsproxy-B)

case "$LABEL" in
  fast)   KCP="fast";   FEC=();                 COMPRESS=() ;;
  normal) KCP="normal"; FEC=();                 COMPRESS=() ;;
  slow)   KCP="slow";   FEC=();                 COMPRESS=() ;;
  fec)    KCP="normal"; FEC=(-fec-data 4 -fec-parity 2); COMPRESS=() ;;
  comp)   KCP="normal"; FEC=();                 COMPRESS=(-compress) ;;
  *) echo "unknown config_label: $LABEL" >&2; exit 1 ;;
esac

PIDS=()
WORK_PIDS=()
SINK_DIR="$(mktemp -d)"

cleanup() {
  kill "${WORK_PIDS[@]}" 2>/dev/null || true
  kill "${PIDS[@]}" 2>/dev/null || true
  rm -rf "$SINK_DIR"
}
trap cleanup EXIT INT TERM

# clear any stale processes from an earlier run (-x: exact name, never the shell)
for p in dnstt-server dnstt-client dnsproxy tcpdump; do
  pkill -9 -x "$p" 2>/dev/null || true
done
pkill -9 -f "http.server $SINK_PORT" 2>/dev/null || true
sleep 1

# content of assorted sizes for the sink to serve
for kb in 1 4 16 64; do
  head -c "$((kb * 1024))" /dev/urandom | base64 > "$SINK_DIR/f${kb}k.txt"
done

# keypair
"$DNSTT_SERVER" -gen-key -privkey-file server.key -pubkey-file server.pub

# shared HTTP sink
( cd "$SINK_DIR" && python3 -m http.server "$SINK_PORT" >/dev/null 2>&1 ) &
PIDS+=($!)
sleep 1

# shared tunnel server (authoritative NS for ZONE) -> local sink
"$DNSTT_SERVER" -udp "127.0.0.1:$TUNNEL_SERVER_PORT" -privkey-file server.key -mtu 1000 \
  "${FEC[@]}" "${COMPRESS[@]}" -kcp-mode "$KCP" \
  "$ZONE" "127.0.0.1:$SINK_PORT" >/dev/null 2>&1 &
PIDS+=($!)

# shared DoQ server (dnsproxy-B): DoQ :853 -> plain DNS :53 for ZONE
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
  -keyout key.pem -out cert.pem -days 30 -nodes \
  -subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1" >/dev/null 2>&1
"$DNSPROXY" -l 127.0.0.1 --quic-port="$RESOLVER_PORT" --udp-buf-size=4096 \
  --tls-crt=cert.pem --tls-key=key.pem \
  -u "[/$ZONE/]127.0.0.1:$TUNNEL_SERVER_PORT" -u 8.8.8.8:53 -p 0 >/dev/null 2>&1 &
PIDS+=($!)
sleep 2

# per-chain: dnsproxy-A_i (plain DNS -> DoQ upstream) + dnstt-client_i
for i in $(seq 0 $((PARALLEL - 1))); do
  dport=$((CLIENT_DNS_BASE + i))
  lport=$((TUNNEL_LOCAL_BASE + i))
  "$DNSPROXY" -l 127.0.0.1 -p "$dport" \
    -u "quic://127.0.0.1:$RESOLVER_PORT" --insecure >/dev/null 2>&1 &
  PIDS+=($!)
  "$DNSTT_CLIENT" -udp "127.0.0.1:$dport" \
    "${FEC[@]}" "${COMPRESS[@]}" -kcp-mode "$KCP" \
    -pubkey-file server.pub "$ZONE" "127.0.0.1:$lport" >/dev/null 2>&1 &
  PIDS+=($!)
done
sleep 3

# capture the DoQ leg (all chains share udp :853)
tcpdump -i lo -w "$OUT" "udp port $RESOLVER_PORT" >/dev/null 2>&1 &
PIDS+=($!)

# per-chain workload: fetch assorted files with randomized think time
FILES=(f1k.txt f4k.txt f16k.txt f64k.txt)
run_workload() {
  local lport="$1"
  local end=$(( $(date +%s) + SECONDS_RUN ))
  while [ "$(date +%s)" -lt "$end" ]; do
    local f="${FILES[$((RANDOM % ${#FILES[@]}))]}"
    curl -s --max-time 8 "http://127.0.0.1:${lport}/${f}" -o /dev/null || true
    sleep "0.$((RANDOM % 9 + 1))"
  done
}
for i in $(seq 0 $((PARALLEL - 1))); do
  run_workload "$((TUNNEL_LOCAL_BASE + i))" &
  WORK_PIDS+=($!)
done

wait "${WORK_PIDS[@]}" 2>/dev/null || true
sleep 1
echo "captured ($PARALLEL chains, ${SECONDS_RUN}s, cfg=$LABEL) -> $OUT"
