#!/usr/bin/env python3
"""Full relay-path diagnostic.

Runs the whole chain with every component logging to a file, then prints all
logs + a curl test, so we can see exactly where the tunnel breaks:

  curl -> dnstt-client (UDP 5300) -> dnsproxy-A -> DoQ 853 -> dnsproxy-B
       -> dnstt-server (UDP 53) -> http.server :8000

Run in Colab:  !cd /content/doq-tunnel-dataset && git pull && python3 src/diagnose_relay.py
"""
import os
import subprocess
import time

BIN = os.path.abspath('bin')
ZONE = 't.example.com'

for pat in ('dnstt', 'dnsproxy', 'http.server', 'tcpdump'):
    subprocess.run(f'pkill -f {pat}', shell=True)

# self-signed cert for the DoQ server (if not present from a prior run)
if not os.path.exists('cert.pem'):
    subprocess.run(
        'openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 '
        '-keyout key.pem -out cert.pem -days 30 -nodes '
        '-subj "/CN=localhost" -addext "subjectAltName=IP:127.0.0.1"',
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def start(name, args):
    f = open(f'/tmp/{name}.log', 'w')
    return subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)

subprocess.run([f'{BIN}/dnstt-server', '-gen-key', '-privkey-file', 'server.key', '-pubkey-file', 'server.pub'])

sink = start('sink', ['python3', '-m', 'http.server', '8000'])
time.sleep(1)
srv = start('server', [f'{BIN}/dnstt-server', '-udp', '127.0.0.1:53', '-privkey-file', 'server.key', '-mtu', '1150', ZONE, '127.0.0.1:8000'])
pxyb = start('proxyB', [f'{BIN}/dnsproxy', '-l', '127.0.0.1', '--quic-port=853', '--udp-buf-size=4096', '--tls-crt=cert.pem', '--tls-key=key.pem', '-u', f'[/{ZONE}/]127.0.0.1:53', '-u', '8.8.8.8:53', '-p', '0'])
pxya = start('proxyA', [f'{BIN}/dnsproxy', '-l', '127.0.0.1', '-p', '5300', '-u', 'quic://127.0.0.1:853', '--insecure'])
time.sleep(3)
cli = start('client', [f'{BIN}/dnstt-client', '-udp', '127.0.0.1:5300', '-pubkey-file', 'server.pub', ZONE, '127.0.0.1:7000'])
time.sleep(3)

print('== curl test (through tunnel) ==')
r = subprocess.run(['curl', '-sv', '--max-time', '8', 'http://127.0.0.1:7000/'],
                   capture_output=True, text=True)
print('curl rc:', r.returncode)
print('stdout:', r.stdout[-400:] if r.stdout else '(empty)')
print('stderr:', r.stderr[-400:] if r.stderr else '(empty)')

for name in ('sink', 'server', 'proxyB', 'proxyA', 'client'):
    p = f'/tmp/{name}.log'
    txt = open(p).read() if os.path.exists(p) else ''
    print(f'== {name}.log (tail) ==')
    print('\n'.join(txt.splitlines()[-15:]) or '(empty)')

for proc in (cli, pxya, pxyb, srv, sink):
    proc.terminate()
