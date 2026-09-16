# DeepGraphAudit

**Graph neural networks for auditing accounting journal entries.**

Reference implementation for:

> Huang, Q., Schreyer, M., Michiles Jr, N. R., & Vasarhelyi, M. A. (2026). [*Connecting the Dots: Graph Neural Networks for Auditing Accounting Journal Entries.*](https://publications.aaahq.org/ajpt/article-abstract/doi/10.2308/AJPT-2024-058/23197/Connecting-the-Dots-Graph-Neural-Networks-for) Auditing: A Journal of Practice & Theory, 1–27.

This is the exact research code used to produce the paper's results — shared for transparency and reproducibility, not polished for retail. It has seen more `print()` statements than unit tests, and it will happily limit itself to 4 CPU threads whether you asked for that or not. Auditors double-check things; so should you before trusting it with your general ledger.

---

## 🕸️ How it works

Each journal entry is turned into a small graph, then embedded and reconstructed by a **graph variational autoencoder**. Entries the model reconstructs poorly, or whose embedding looks unusual, are flagged for audit follow-up.

```
journal entry               entry graph                     graph-VAE                     scoring
┌───────────────┐   nodes = accounts posted to      ┌───────────────────┐        reconstruction error
│  line items   │ ─ edges = debit → credit ───────▶ │ encode → z (2D) →  │ ─────▶ or outlier score on z
│  (accounts,   │   features = account/entry attrs  │       decode       │        (IForest / LOF / OCSVM /
│  amounts, …)  │                                    └───────────────────┘        HDBSCAN / …)
└───────────────┘
```

- **Dynamic mode** (default): each entry keeps its own (variable) number of accounts as graph nodes.
- **Static mode**: entries are zero-padded to the full chart of accounts.
- **Baseline**: a non-graph, flat-feature autoencoder for comparison (`-mode baseline`).
- **Evaluation**: synthetic *global* and *local* anomalies are injected into the data, and detection is scored via ROC-AUC / PR-AUC / precision / recall / F1.

## 🗂️ Repository layout

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point and experiment configuration |
| `DataHandler/` | Loading, cleaning, feature engineering, adjacency/feature-matrix construction |
| `AnomalyHandler/` | Synthetic anomaly injection + latent-space anomaly detection (PyOD, HDBSCAN) |
| `ModelHandler/` | Graph-VAE (`GraphConvLayer`, `GNNEncoder`, `GNNAutoencoder*`) and the flat-feature baseline |
| `ExperimentHandler/` | Training/validation loops (`GraphAutoencoderExperiment{Static,Dynamic}`, `AutoencoderExperimentDynamic`) |
| `EvaluationHandler/` | Scoring metrics: ROC-AUC / PR-AUC / accuracy / precision / recall / F1 |
| `GridSearchHandler/` | Hyperparameter sweeps (`beta`, feature-embedding dim, seed) |
| `VisualisationHandler/` | Static + interactive (Plotly) embedding plots, NetworkX entry-graph renderings |
| `LoggingHandler/` | CSV experiment logs + optional Weights & Biases tracking |
| `UtilsHandler/` | Experiment-directory bootstrapping, checkpointing, arg-parsing helpers |
| `100_datasets/` | Input data goes here (not included — see the Data section below) |
| `200_experiments/` | Experiment outputs (created automatically) |

## 📦 Installation

Python 3.8+, then:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions this was verified against; loosen them if they clash with your setup. `wandb` is only needed if you run with `-wandb True` (default) — pass `-wandb False` to skip creating an account. GPU is used automatically when available; CPU works too, just bring snacks.

## 📊 Data

| `-dataset` | Source |
|---|---|
| `ey` | [EY Academic Resource Center (EYEARC)](https://www.ey.com/en_us/about-us/ey-foundation-and-university-relations/academic-resource-center) "Analytics Mindset" teaching case — the only dataset a reader can source independently 🎓 |
| `sap` | Confidential SAP ERP extract from an audit engagement 🔒 |

**No data is included in this repository** — the SAP extract is confidential engagement data under NDA. Populate `100_datasets/` yourself; if your export uses different columns, adjust the mapping in the corresponding dataset block of `main.py` and add matching preprocessing methods in `DataHandler/DataHandler.py`.

## 🚀 Quickstart

```bash
# graph autoencoder on the EY dataset
python main.py -dataset ey -mode dynamic -experiment graph_autoencoder \
  -train_iterations 500 -feat_embed_dim 12 -beta 0.5 -algo iforest -wandb False

# non-graph baseline for comparison
python main.py -dataset ey -mode baseline -wandb False
```

Add `-sample_eval True -sample_size 501` to smoke-test on a small slice of the data before a full run.

## 🎛️ Key configuration flags

Full list with defaults: `python main.py -h`.

| Group | Flags |
|---|---|
| Experiment | `-mode` (`dynamic`/`static`/`baseline`), `-dataset`, `-grid`, `-data_dir`, `-base_dir` |
| Architecture | `-encoder_dim`, `-decoder_dim`, `-feat_embed_dim`, `-lat_embed_dim`, `-encoder_bottleneck`/`-decoder_bottleneck` |
| Training | `-train_iterations`, `-train_batch_size`, `-loss` (`mse`/`bce`), `-learning_rate`, `-beta`, `-seed` |
| Anomaly detection | `-algo` (`iforest`/`lof`/`ocsvm`/`hdbscan`/`knn`/`hbos`) and its parameters |
| Grid search | `-grid_seed`, `-grid_beta`, `-grid_feat_embed_dim`, `-grid_min_cluster_size`, … |

## 🧾 Outputs

Each run writes a timestamped, parameter-encoded folder under `200_experiments/` with:

`00_param/` config · `01_statistics/` loss & metric logs · `02_results/` entries enriched with embeddings (`z1`,`z2`), reconstruction error, and anomaly score · `03_visualizations/` static + interactive embedding plots.

## 🔁 Reproducing the paper's grid search

```bash
bash start_ey_grid_search.sh   # sweeps beta, feat_embed_dim, seed over the EY dataset
bash start_sap_grid_search.sh  # sweeps beta, learning rate, seed over the SAP dataset
```

Edit the array variables at the top of each script to match your compute budget.

## 📝 Citation

```bibtex
@article{huang2026connecting,
  title   = {Connecting the Dots: Graph Neural Networks for Auditing Accounting Journal Entries},
  author  = {Huang, Qiao and Schreyer, Marco and Michiles Jr, Norton R. and Vasarhelyi, Miklos A.},
  journal = {Auditing: A Journal of Practice \& Theory},
  year    = {2026},
  pages   = {1--27}
}
```

## ⚖️ License

[BSD 3-Clause](LICENSE)
