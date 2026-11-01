#!/usr/bin/env bash
# Generate malicious DoQ tunneling traffic (dnstt tunneled over DoQ via dnsproxy).
#
# Topology (all loopback):
#   curl -> dnstt-client (plain UDP 5300) -> dnsproxy-A (plain DNS -> DoQ)
#        -> dnsproxy-B (DoQ server 853 -> plain UDP 53) -> dnstt-server
#        -> local TCP sink :8000
#
# The DoQ transport is handled by dnsproxy on both ends (well-tested), NOT by
# dnstt's DoQ client (which produced "bad certificate" / "bad rdata" errors).
# We capture the DoQ leg on UDP 853.
#
# Usage: ./generate_malicious.sh <output.pcap> [config_label] [seconds]
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
CLIENT_DNS_PORT=5300      # dnsproxy-A plain DNS listen port
RESOLVER_PORT=853         # DoQ port (dnsproxy-A -> dnsproxy-B)
CLIENT_LOCAL_PORT=7000

case "$LABEL" in
  fast)   KCP="fast";   FEC=();                 COMPRESS=() ;;
  normal) KCP="normal"; FEC=();                 COMPRESS=() ;;
  slow)   KCP="slow";   FEC=();                 COMPRESS=() ;;
  fec)    KCP="normal"; FEC=(-fec-data 4 -fec-parity 2); COMPRESS=() ;;
  comp)   KCP="normal"; FEC=();                 COMPRESS=(-compress) ;;
  *) echo "unknown config_label: $LABEL" >&2; exit 1 ;;
esac

cleanup() {
  kill "$CAPTURE_PID" "$CLIENT_PID" "$PROXY_A_PID" "$PROXY_B_PID" "$SERVER_PID" "$SINK_PID" 2>/dev/null || true
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

# 2) server-side dnsproxy: DoQ server (853) -> forwards ZONE to dnstt-server
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
  -keyout key.pem -out cert.pem -days 30 -nodes \
  -subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1" >/dev/null 2>&1
"$DNSPROXY" -l 127.0.0.1 --quic-port="$RESOLVER_PORT" --udp-buf-size=4096 \
  --tls-crt=cert.pem --tls-key=key.pem \
  -u "[/$ZONE/]127.0.0.1:$TUNNEL_SERVER_PORT" -u 8.8.8.8:53 -p 0 &
PROXY_B_PID=$!

# 3) client-side dnsproxy: plain DNS (5300) -> DoQ upstream (853)
"$DNSPROXY" -l 127.0.0.1 -p "$CLIENT_DNS_PORT" \
  -u "quic://127.0.0.1:$RESOLVER_PORT" --insecure &
PROXY_A_PID=$!
sleep 2

# 4) tunnel client -> plain DNS to dnsproxy-A (which relays it over DoQ)
"$DNSTT_CLIENT" -udp "127.0.0.1:$CLIENT_DNS_PORT" \
  "${FEC[@]}" "${COMPRESS[@]}" -kcp-mode "$KCP" \
  -pubkey-file server.pub "$ZONE" "127.0.0.1:$CLIENT_LOCAL_PORT" &
CLIENT_PID=$!
sleep 2

# 5) capture the DoQ leg (dnsproxy-A <-> dnsproxy-B)
tcpdump -i lo -w "$OUT" "udp port $RESOLVER_PORT" >/dev/null 2>&1 &
CAPTURE_PID=$!

# 6) push data through the tunnel
END=$((SECONDS + SECONDS_RUN))
while [ "$SECONDS" -lt "$END" ]; do
  curl -s --max-time 5 "http://127.0.0.1:$CLIENT_LOCAL_PORT/" -o /dev/null || true
  sleep 1
done

sleep 1
echo "captured -> $OUT"
