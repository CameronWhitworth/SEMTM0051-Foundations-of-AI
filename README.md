# SEMTM0051 Foundations of AI — Coursework

Code and report files for the **Underfitting, Overfitting and Held-out Data** coursework.

## Files

* `solution.py` — Combined script running Exercises 1–8 and saving figures to `images/`.
* `prep_football.py` — Cleans `football_raw.csv` (filtering `90s >= 3.0`, subsampling to 120 players, and standardising `Sh/90` and `SoT/90`) into `football_data.csv`.
* `fit.py` and `make_data.py` — Coursework helper modules for model training, evaluation, and synthetic data generation.
* `report.tex` — LaTeX source for the PDF report (up to 6 pages for Exercises 1–7 and up to 3 pages for Exercise 8).

## How to Run

1. Generate the main coursework dataset if not already present:
   ```bash
   python make_data.py
