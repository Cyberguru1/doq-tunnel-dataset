#!/usr/bin/env bash
# Fetch the labeled DoQ tunneling dataset.
#
# The raw pcaps and combined feature CSV are large and are hosted separately
# (Zenodo / IEEE DataPort, CC BY 4.0), not in this repo.
#
# TODO: set the published DOI/URL once the dataset is released.
set -euo pipefail

echo "Dataset download is not configured yet."
echo "Reproduce it locally with:"
echo "  bash src/generate_malicious.sh <out.pcap> <config_label> <seconds>"
echo "  bash src/generate_benign.sh <out.pcap> <seconds>"
echo "  python3 src/merge_dataset.py --config sources.json --out combined.csv"
exit 1
