# DoQ Tunneling Detection — Pipeline & Dataset Tooling

Reproducible pipeline for detecting DNS tunneling over **DNS-over-QUIC (DoQ)**:
generate labeled DoQ traffic, extract a unified 28-feature flow representation,
and train classifiers. Companion to the DoQ tunneling dataset (published separately).

## Contents

- `src/extract_features.py` — unified 28-feature extractor (pcaps and IEEE DoQ+QUIC CSVs)
- `src/merge_dataset.py` — merge malicious + benign + non-DoQ into one labeled CSV
- `src/generate_malicious.sh` — dnstt-fork `-doq` loopback tunnel generator + capture
- `src/generate_benign.sh` — benign DoQ generator + capture
- `notebooks/doq_pipeline.ipynb` — single sectioned Colab notebook (setup → generate → extract → train)
- `data/download.sh` — fetch the published dataset (once the DOI is live)

## Quick start — Google Colab

1. In Colab: **File → Open notebook → GitHub**, search for this repo, and open
   `notebooks/doq_pipeline.ipynb`.
2. Run the sections top to bottom. `REPO_URL` is pre-set; edit `IEEE_PATH` in
   Section 3 to point at your unzipped DoQ+QUIC packet CSVs.

## Quick start — local

```bash
python3 -m pip install dnspython aioquic scapy pandas scikit-learn xgboost
# place dnsproxy + dnstt-server/dnstt-client in ./bin first

bash src/generate_malicious.sh malicious_doq_fast.pcap fast 120
bash src/generate_benign.sh benign_doq_lab.pcap 120
python3 src/merge_dataset.py --config sources.json --out combined.csv
```

## Data & citation

The labeled DoQ tunneling dataset is hosted separately (Zenodo / IEEE DataPort,
CC BY 4.0); `data/download.sh` will fetch it once released. Methodology and the
citation bundle live in `METHODOLOGY_CITATIONS.md` / `EXECUTION_PLAN.md` (project root).

## License

- Code: MIT (see `LICENSE`)
- Dataset (published separately): CC BY 4.0
