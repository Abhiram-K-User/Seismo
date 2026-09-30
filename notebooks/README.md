# notebooks/

Exploratory analysis. Reusable code lives in `src/seismic_ml/`; notebooks import it.

| Notebook | Phase | Status | Purpose |
|---|---|---|---|
| `01_data_exploration.ipynb` | 1 | ✅ executed on the PNW data | Dataset overview, labels, multi-station coverage, sampling/windows, 3-component validation, station/event distributions, confounders, waveform inspection, data quality, OOD feasibility, split-design inputs |
| `02_waveform_analysis.ipynb` | 2 | ✅ executed on the PNW subsets | Spectra by instrument (BH cut-off), spectral SNR, signal duration vs magnitude, before/after preprocessing, alignment/mask/normalisation checks, S1 split summary |
| `03_feature_analysis.ipynb` | 3–4 | placeholder | Learned embeddings and time-frequency features |

## Running

```bash
pip install -e ".[notebooks]"
python scripts/download_data.py      # once; add --data-root to store data elsewhere
jupyter lab notebooks/               # or open in VS Code / Colab
```

The setup cell locates the project root automatically and reads the data root from
`SEISMIC_DATA_ROOT` / `configs/config.yaml` (see `data/README.md`); a commented
Colab + Google Drive block is included. No machine-specific path appears in any notebook.

Headless re-execution (refreshes the stored outputs, UTF-8 safe):

```bash
python scripts/run_notebooks.py notebooks/01_data_exploration.ipynb notebooks/02_waveform_analysis.ipynb
```

Do **not** use `jupyter-execute --inplace` on Windows: it reads notebooks with the
cp1252 code page and double-encodes all non-ASCII text on save (this happened to
notebook 01 in Phase 1 and was repaired in Phase 2; a test now guards against it).

## Rules

- Notebooks explore; reusable code belongs in `src/seismic_ml/`.
- Import project code (`from seismic_ml...`) instead of copying it.
- `01_data_exploration.ipynb` and `02_waveform_analysis.ipynb` are committed **with
  outputs**, because their figures document the findings and parameter choices in
  `PROJECT_CONTEXT.md`. Other notebooks: clear
  outputs before committing unless they document a recorded finding.
- Outputs must never print machine-specific absolute paths (a test scans for them).
- Findings that drive decisions go into `PROJECT_CONTEXT.md` (Decision Log),
  not only into a notebook.
- Label every statement as observed, inferred, documented or unknown; never
  present synthetic data as real data or report model results here.
