# Making Sound Classification More Reliable Across Microphones

Group 10's final project for CISC 473 (Fall 2026), topic P19. Masih Hashemi, Anand Swamy and Kiarash Mirkamandari.

We tested whether removing "microphone directions" from frozen AST embeddings (our method, CPCP) helps an ESC-50 classifier on filters it hasn't seen, compared to just training on the filtered clips. Short answer: it doesn't beat plain augmentation. The notebook also has the UrbanSound8K baseline, a mel-CNN, SpecAugment, a small AST fine-tune, a patch-size study, an attention check and UrbanSound8K to ESC-50 transfer.

Everything is in one notebook, `CISC473_CPCP.ipynb`.

## Setup

We ran it with Python 3.11 on a Mac, CPU only.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Linux, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the pip command to get the CPU version of PyTorch.

## Running it

```
jupyter nbconvert --to notebook --execute --inplace CISC473_CPCP.ipynb --ExecutePreprocessor.timeout=-1
```

Or open the notebook and run all the cells from the top.

The notebook downloads ESC-50 and the AST checkpoint on its own. UrbanSound8K is about 6 GB from Zenodo. If that download keeps failing, section 7 explains how to put the archive in place yourself.

Results are saved in `cisc473_audio_three_seeds/` next to the notebook. You can point it somewhere else with the `CISC473_THREE_SEED_ROOT` environment variable. If a run stops halfway, running it again picks up where it left off. Each experiment can be switched on or off with the `RUN_*` flags in section 2, but changing any setting there starts a new results folder.

The final tables end up in `final_three_seed/<run id>/`. The main ones are `esc50/summary_percent.csv` and `esc50/paired_bootstrap.json`, and `report_bundle.zip` has everything we used in the report.

## Results

ESC-50, frozen AST, five folds, mean ± SD over seeds 42, 123 and 999:

| Method | Clean accuracy (%) | Held-out filter macro F1 (%) |
|---|---|---|
| Original only | 95.03 ± 0.18 | 94.71 ± 0.12 |
| Augmentation only | 95.43 ± 0.18 | 94.93 ± 0.04 |
| Unprotected projection | 95.50 ± 0.08 | 95.07 ± 0.09 |
| CPCP | 95.38 ± 0.10 | 94.90 ± 0.10 |

CPCP minus augmentation on filtered macro F1 was -0.023 points (95% paired bootstrap interval -0.168 to +0.131).
