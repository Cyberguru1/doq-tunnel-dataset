#!/usr/bin/env python3
"""Diagnose the DoQ tunnel end-to-end.

Run from the repo root in Colab (after `git pull`):

    !cd /content/doq-tunnel-dataset && python3 src/diagnose_tunnel.py

Tests the two most likely failure points and prints the component logs.
"""
import os
import subprocess
import time

import dns.message
import dns.query

BIN = os.path.abspath('bin')
ZONE = 't.example.com'

# clean any leftovers
for pat in ('dnstt', 'dnsproxy', 'http.server'):
    subprocess.run(f'pkill -f {pat}', shell=True)

sink = subprocess.Popen(
    ['python3', '-m', 'http.server', '8000'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
srv = subprocess.Popen(
    [f'{BIN}/dnstt-server', '-udp', '127.0.0.1:53', '-privkey-file', 'server.key',
     ZONE, '127.0.0.1:8000'],
    stdout=open('/tmp/dnstt-server.log', 'w'), stderr=subprocess.STDOUT,
)
pxy = subprocess.Popen(
    [f'{BIN}/dnsproxy', '-l', '127.0.0.1', '--quic-port=853',
     '--tls-crt=cert.pem', '--tls-key=key.pem',
     '-u', f'[/{ZONE}/]127.0.0.1:53', '-u', '8.8.8.8:53', '-p', '0'],
    stdout=open('/tmp/dnsproxy.log', 'w'), stderr=subprocess.STDOUT,
)
time.sleep(4)

q = dns.message.make_query(ZONE, 'SOA')

print('== TEST 1: dnstt-server direct (UDP 53) ==')
try:
    r = dns.query.udp(q, '127.0.0.1', port=53, timeout=3)
    print('  rcode:', r.rcode(), '| answers:', len(r.answer))
except Exception as e:
    print('  FAILED:', repr(e))

print('== TEST 2: via dnsproxy DoQ (UDP 853) ==')
try:
    r2 = dns.query.quic(q, '127.0.0.1', port=853, verify=False, timeout=3)
    print('  rcode:', r2.rcode(), '| answers:', len(r2.answer))
except Exception as e:
    print('  FAILED:', repr(e))

print('== dnstt-server.log (tail) ==')
print('\n'.join(open('/tmp/dnstt-server.log').read().splitlines()[-20:]) or '(empty)')
print('== dnsproxy.log (tail) ==')
print('\n'.join(open('/tmp/dnsproxy.log').read().splitlines()[-20:]) or '(empty)')

for p in (srv, pxy, sink):
    p.terminate()
