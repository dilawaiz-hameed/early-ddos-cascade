# Lightweight Early-Stage DDoS Detection: Phase 3 Implementation

Computer Networks research project (FAST-NUCES). Authors: Dilawaiz Hameed (i232518) and Mariam Shaiq (i233250).

This repository implements the Phase 2 methodology:
- an early-window, confidence-gated cascade detector;
- an NSGA-II search over accuracy, time-to-detect and resource cost;
- the baselines, generalisation tests, cost benchmark, ablations and error analysis reported in the Phase 3 report.

## Requirements

- Python 3.10+ with `numpy`, `scipy`, `pandas`, `scikit-learn` (tested with 1.9.1) and `matplotlib`.
- No packet library is needed: `src/pcapio.py` reads pcap and pcapng directly.

## Data (all public)

| Folder | Source | Role |
|---|---|---|
| `data/cic2019/` | CIC-DDoS2019 sample PCAPs shipped with LUCID: https://github.com/doriguzzi/lucid-ddos (`sample-dataset/`) | Main dataset (train/val/test). Contains benign traffic plus DNS reflection, SYN flood, UDP-Lag and WebDDoS attacks. |
| `data/stopddos/` | StopDDoS packet-capture collection: https://github.com/StopDDoS/packet-captures | External attack traffic (real-world and lab DDoS captures, one-way). |
| `data/sanders/` | Practical Packet Analysis captures: https://github.com/chrissanders/packets | External benign traffic. Malicious captures are excluded in `flows.py`. |
| `data/malcolm/` | Malcolm protocol samples: https://github.com/mmguero-dev/Malcolm-PCAP (`pcaps/protocols/`: HTTP_1, DNS, SSL, QUIC, SSH, FTP, SMTP, Syslog, RDP, SIP, TFTP, WireGuard) | External benign traffic. |

Clone those repositories and copy the files into the folders above.

## Run everything

```bash
python src/flows.py                      # pcaps -> flows + first-32-packet tables (data/flows_*.npz)
python src/run_search.py 0 1 2           # Experiment E: NSGA-II vs random search vs GA (3 seeds, 800 evals each)
python src/run_experiments.py            # baselines, Experiments A-C, ablations, error analysis
OMP_NUM_THREADS=1 taskset -c 0 python src/cost_bench.py   # Experiment D: cost on one pinned core
python src/make_figures.py               # all figures -> figures/
```

## Outputs

Results are written to `results/`:
- `experiments.json`: every number in the report;
- `cost_bench.json`: the cost benchmark;
- `search_*.json`: every configuration the searches evaluated.

## Code map

| File | Purpose |
|---|---|
| `src/pcapio.py` | Dependency-free pcap/pcapng reader (Ethernet, VLAN, raw IP, Linux SLL/SLL2). |
| `src/flows.py` | Bidirectional flow assembly (60 s idle timeout) and labelling. |
| `src/features.py` | 31 early-window features in four semantic groups, each tagged with how many packets it needs. A window (N, T) closes at the N-th packet or T seconds, whichever comes first. |
| `src/data.py` | Feature cache, decision times, chronological 60/20/20 split per capture with time blocks. |
| `src/models.py` | Model zoo, cost measurement and the two-stage `Cascade`. |
| `src/search.py` | NSGA-II, random search, single-objective GA, Pareto front and exact hypervolume. |
| `src/metrics.py` | Macro-F1, MCC, PR-AUC, prevalence re-weighting, time-to-detect, block bootstrap, Wilcoxon + Holm. |
| `src/run_experiments.py` | All held-out evaluations. |
| `src/cost_bench.py` | Latency, throughput and size on one core. |
| `src/make_figures.py` | Figures. |

## Full CIC-DDoS2019 captures (optional, larger run)

1. Put the selected capture files in `PCAP-01-12_0-0249` (1 Dec 2018, training day) and `PCAP-03-11` (3 Nov 2018, testing day).
2. Run `run_full.bat` on Windows. It runs the steps below with these settings: `CN_MAIN=cicfull`, `CN_RESULTS=results_full`, `CN_FIGURES=figures_full`.
   - `src/flows_full.py`: builds flows from the captures.
     - Labels: attacker 172.16.0.5 = attack; the attack type comes from the capture file number (table in the script).
     - Memory: files are streamed one at a time; 1 in 20 attack flows is sampled; at most 1,000 attack and 5,000 benign flows are kept per file.
   - The other scripts then run with a day split: train and validate on 1 Dec, test on all of 3 Nov.
3. `CN_POP` and `CN_GENS` change the search budget (default 32 and 24).

`src/peek_pcaps.py <folder>` prints, for each capture file, its start time, the share of attacker traffic and its top ports.
