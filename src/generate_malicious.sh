#!/usr/bin/env bash
# Generate malicious DoQ tunneling traffic (dnstt-fork -doq) and capture it.
#
# Fully loopback / offline: no public domain or external resolver required.
# Topology:
#   curl -> dnstt-client (TCP 7000) -> DoQ -> dnsproxy resolver (UDP 853)
#        -> UDP 53 -> dnstt-server (authoritative NS) -> local TCP sink :8000
#
# Usage:
#   ./generate_malicious.sh <output.pcap> [config_label] [seconds]
#
# config_label: fast | normal | slow | fec | comp
#
# Prerequisites (put binaries in ./bin or set BIN_DIR):
#   - dnstt-server / dnstt-client built from https://github.com/IvanTopGaming/dnstt
#   - dnsproxy from https://github.com/AdguardTeam/dnsproxy/releases
#   - tcpdump, curl, openssl, python3
set -euo pipefail

OUT="${1:?usage: generate_malicious.sh <output.pcap> [config_label] [seconds]}"
LABEL="${2:-fast}"
SECONDS_RUN="${3:-120}"
BIN_DIR="${BIN_DIR:-./bin}"

DNSTT_SERVER="$BIN_DIR/dnstt-server"
DNSTT_CLIENT="$BIN_DIR/dnstt-client"
DNSPROXY="$BIN_DIR/dnsproxy"

ZONE="t.example.com"
SINK_PORT=8000
TUNNEL_SERVER_PORT=53
RESOLVER_PORT=853
CLIENT_LOCAL_PORT=7000

case "$LABEL" in
  fast)   KCP="fast";   FEC=();         COMPRESS=() ;;
  normal) KCP="normal"; FEC=();         COMPRESS=() ;;
  slow)   KCP="slow";   FEC=();         COMPRESS=() ;;
  fec)    KCP="normal"; FEC=(-fec-data 4 -fec-parity 2); COMPRESS=() ;;
  comp)   KCP="normal"; FEC=();         COMPRESS=(-compress) ;;
  *) echo "unknown config_label: $LABEL" >&2; exit 1 ;;
esac

cleanup() {
  kill "$CAPTURE_PID" "$CLIENT_PID" "$RESOLVER_PID" "$SERVER_PID" "$SINK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 0) keypair
"$DNSTT_SERVER" -gen-key -privkey-file server.key -pubkey-file server.pub

# 1) tunnel server (authoritative NS for ZONE) -> local TCP sink
python3 -m http.server "$SINK_PORT" >/dev/null 2>&1 &
SINK_PID=$!
sleep 1
"$DNSTT_SERVER" -udp "127.0.0.1:$TUNNEL_SERVER_PORT" -privkey-file server.key -mtu 1150 \
  "${FEC[@]}" "${COMPRESS[@]}" -kcp-mode "$KCP" \
  "$ZONE" "127.0.0.1:$SINK_PORT" &
SERVER_PID=$!

# 2) local DoQ recursive resolver (dnsproxy) routing ZONE to the tunnel server
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
  -keyout key.pem -out cert.pem -days 30 -nodes \
  -subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1" >/dev/null 2>&1
"$DNSPROXY" -l 127.0.0.1 --quic-port="$RESOLVER_PORT" --udp-buf-size=4096 \
  --tls-crt=cert.pem --tls-key=key.pem \
  -u "[/$ZONE/]127.0.0.1:$TUNNEL_SERVER_PORT" -u 8.8.8.8:53 -p 0 &
RESOLVER_PID=$!
sleep 2

# 3) tunnel client -> DoQ to the local resolver (pin self-signed cert)
PIN="SHA256:$(openssl x509 -in cert.pem -outform DER | openssl dgst -sha256 -hex | awk '{print $NF}')"
"$DNSTT_CLIENT" -doq "127.0.0.1:$RESOLVER_PORT" -pin-cert "$PIN" -pin-cert-skip-chain \
  "${FEC[@]}" "${COMPRESS[@]}" -kcp-mode "$KCP" \
  -pubkey-file server.pub "$ZONE" "127.0.0.1:$CLIENT_LOCAL_PORT" &
CLIENT_PID=$!
sleep 2

# 4) capture only the DoQ leg (client <-> resolver)
tcpdump -i lo -w "$OUT" "udp port $RESOLVER_PORT" >/dev/null 2>&1 &
CAPTURE_PID=$!

# 5) push real data through the tunnel (sustained, bidirectional)
END=$((SECONDS + SECONDS_RUN))
while [ "$SECONDS" -lt "$END" ]; do
  curl -s --max-time 5 "http://127.0.0.1:$CLIENT_LOCAL_PORT/" -o /dev/null || true
  sleep 1
done

sleep 1
echo "captured -> $OUT"
