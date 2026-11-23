# tie-keepers

**Code and results for:** Christian R. Klein and Reinhard Klein, *Artificial intelligence turns investment in people into cooperation with the future* (submitted).

Much of what people invest in one another happens as a by-product of joint work. When artificial intelligence (AI) takes over this work, relationships must be maintained deliberately, as an investment in cooperation with the future. This repository contains a game-theoretic network model of that situation. In the model, a relationship is maintained only if b/c > (1 − D)(1 + max(1/w, K/c)/T), maintenance becomes a stag hunt, and network cohesion collapses once too few people keep a third relationship. The networks hold together only as long as enough people remain *tie keepers*.

## Quick start

```bash
git clone https://github.com/ChristianRKlein/tie-keepers.git
cd tie-keepers
pip install -r requirements.txt
python run_all.py
```

On Windows you can also double-click `run_all.bat`; in Spyder, open `run_all.py` and press F5.

| Command | What it does | Time (laptop) |
|---|---|---|
| `python run_all.py` | all simulations and all figures | about 2–3 h |
| `python run_all.py --figures` | redraw the figures from the stored results | < 1 min |
| `python run_all.py --quick` | fast test with fewer seeds | about 30 min |

The results used in the paper are already included (`results_S1` … `results_S7`), so `--figures` reproduces all figures immediately. A full run regenerates them; all random seeds are fixed. **Note:** `--quick` overwrites the stored results with test values; restore them with `git checkout -- results_S*`.

Requirements: Python ≥ 3.7 with NumPy ≥ 1.17, pandas, Matplotlib ≥ 3.3 and NetworkX (see `requirements.txt`). The stored results were produced with Python 3.13.

## Structure

| File | Content | Output |
|---|---|---|
| `netzmodell.py` | Stage A: strategic network formation with AI (skills, AI substitution φ(α), pairwise stability); yields the by-product share D(α) and the initial networks | – |
| `zukunft.py` | Stage B: cooperating with the future (deliberate maintenance, trust, decay, rebuilding), the rule and its mean-field prediction | – |
| `struktur.py` | network tools: largest component, coalescence times, σ rule | – |
| `schritt1_ohtsuki.py` | Step 1: cooperation conditions on the model networks (Ohtsuki et al. 2006; Allen et al. 2017) | `results_S1` |
| `schritt2_vierfeldertafel.py` | Step 2: task-level game, human vs AI (Supplementary Note 1) | `results_S2` |
| `schritt3_zukunft.py` | Step 3: main simulations, variants, hysteresis, shared costs | `results_S3` |
| `schritt4_regel.py` | Step 4: rule vs simulation, binding agreements, contacts at distance three | `results_S4` |
| `schritt5_versaeumnis_lhs.py` | Step 5: single-lapse test and Latin hypercube (200 parameter combinations) | `results_S5` |
| `schritt6_review.py` | Step 6: which links are lost, heterogeneous networks and costs, new joint work, long runs | `results_S6` |
| `schritt7_review2.py` | Step 7: degree-preserving rewiring, mapping between output share and by-product share, 20 replicates, network size up to N = 2,000 | `results_S7` |
| `make_figures.py` | all figures of the paper | `paper_figs` |
| `run_all.py` | runs everything in order | `run_all_log.txt` |

## Figures and source data

| Figure | Data |
|---|---|
| Fig. 1 | `results_S3/S3_abhaengigkeit.csv` |
| Fig. 2 | `results_S3/S3_varianten.csv`, `results_S5/S5_versaeumnis.csv` |
| Fig. 3 | `results_S3/S3_varianten.csv`, `results_S4/S4_phasendiagramm.csv`, `results_S5/S5_lhs200.csv` |
| Fig. 4 | `results_S3/S3_hysterese_K.csv` |
| Fig. 5 | `results_S4/S4_abstimmung.csv`, `results_S3/S3_varianten.csv`, `results_S3/S3_subvention.csv` |
| Supplementary Fig. 1 | `results_S4/S4_drei_schritte.csv`, `results_S5/S5_lhs200.csv`, `results_S5/S5_prcc200.csv` |
| Supplementary Fig. 2 | `results_S6/*.csv` |
| Supplementary Fig. 3 | `results_S7/*.csv` |
| Supplementary Note 1 | `results_S1`, `results_S2` |

The code was developed in German; file names, comments and some column names are German (e.g. *Zusammenhalt* = cohesion, *Beziehungen* = relationships per person, *Abhängigkeit* = by-product share D, *Kollaps* = collapse, *Versäumnis* = lapse, *Kontrolle* = control). The variable `lcc` is the share of individuals in the largest connected component.

## Default parameters (Stage B)

| Symbol | Meaning | Value |
|---|---|---|
| μ | maintenance cost per relationship and month without by-product | 0.015 |
| p | probability of a moment of need per month | 0.1 |
| B, k₀ | maximum and saturation of the network value | 1, 4 |
| η | weight of contacts at distance two | 0.25 |
| β (T) | discount factor (horizon T = 1/(1 − β)) | 0.95 (20 months) |
| w, d₀ | decay with one-sided / no maintenance | 0.25, 0.30 |
| K | rebuilding cost | 0.1 |
| r | probability of considering a rebuild | 0.1 |
| ε | error rate | 0.01 |
| N | individuals | 300 |

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT License, see `LICENSE`.

## Contact

Christian R. Klein, University Hospital Bonn — christian.klein@ukbonn.de
