#!/usr/bin/env bash
# Generate benign DoQ traffic (local dnsproxy DoQ resolver + dnspython[doq]
# client) and capture it. Fully loopback; self-signed cert with verify=False.
#
# Usage:
#   ./generate_benign.sh <output.pcap> [seconds]
#
# Prerequisites: dnsproxy binary (./bin/dnsproxy or set BIN_DIR),
# tcpdump, openssl, python3 with dnspython[doq] installed.
set -euo pipefail

OUT="${1:?usage: generate_benign.sh <output.pcap> [seconds]}"
SECONDS_RUN="${2:-120}"
BIN_DIR="${BIN_DIR:-./bin}"
DNSPROXY="$BIN_DIR/dnsproxy"
RESOLVER_PORT=853

cleanup() {
  kill "$CAPTURE_PID" "$RESOLVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# DoQ resolver on loopback
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 \
  -keyout key.pem -out cert.pem -days 30 -nodes \
  -subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1" >/dev/null 2>&1
"$DNSPROXY" -l 127.0.0.1 --quic-port="$RESOLVER_PORT" \
  --tls-crt=cert.pem --tls-key=key.pem -u 8.8.8.8:53 -p 0 &
RESOLVER_PID=$!
sleep 2

tcpdump -i lo -w "$OUT" "udp port $RESOLVER_PORT" >/dev/null 2>&1 &
CAPTURE_PID=$!

# benign DoQ query generator (dnspython[doq]; verify=False for the self-signed cert)
python3 - "$RESOLVER_PORT" "$SECONDS_RUN" <<'PY'
import sys, time
import dns.message, dns.query

port = int(sys.argv[1])
secs = int(sys.argv[2])
q = dns.message.make_query("example.com", "A")
end = time.time() + secs
while time.time() < end:
    try:
        dns.query.quic(q, "127.0.0.1", port=port, verify=False, timeout=2)
    except Exception:
        pass
    time.sleep(0.05)
PY

sleep 1
echo "captured -> $OUT"
