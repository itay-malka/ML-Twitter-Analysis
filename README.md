# ML Course – Twitter Sentiment Analysis

Multi-class sentiment classification (Positive / Negative / Neutral / Irrelevant) on tweets,
using a Naive Bayes classifier.

## Project structure

```
ml_course_twitter_analysis/
├── twitter_sentiment_analysis.ipynb   # main notebook
├── twitter_training.csv               # training data
├── twitter_validation.csv             # validation/test data
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/itay-malka/ML-Twitter-Analysis.git
cd ML-Twitter-Analysis
```

### 2. Create and activate a virtual environment

A `.venv` folder is already present in this project. If you need to create it from scratch:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your terminal prompt once it's active.

### 3. Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

### 4. Register the venv as a Jupyter kernel (recommended)

This ensures the notebook actually uses the packages you just installed, instead of some other Python install:

```bash
python -m ipykernel install --user --name=ml_twitter_venv --display-name "Python (ml_twitter_venv)"
```

## Running the project

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
   or, if you're using VS Code, just open `twitter_sentiment_analysis.ipynb` there directly.

2. In the notebook, make sure the selected kernel (top-right in Jupyter, or the kernel picker in VS Code) is **"Python (ml_twitter_venv)"** — not a system/global Python install.

3. Run all cells top to bottom (**Run → Run All Cells**, or `Kernel → Restart Kernel and Run All`).

The notebook expects `twitter_training.csv` and `twitter_validation.csv` to sit in the **same folder** as the notebook itself (no path changes needed if you keep the structure above).

## What the notebook does

The notebook is organised to mirror the assignment's sections one-to-one. Explanations are
written in Hebrew (the language of the submission video); all code and comments are in English.

| Section | Assignment part | Contents |
|---|---|---|
| 1 | Part 1 (5 pts) | Loads both provided CSVs unsplit, shows the first 5 rows of each, and profiles the class balance in train vs. test. |
| 2 | Part 2 (35 pts) | Stateless `clean_tweet` normalisation (lower-casing, URL / @mention removal, letters-only filtering) followed by a stateful vectoriser fitted on the training set only. Demonstrated before/after on 3 train **and** 3 test examples. |
| 3 | Part 3 (35 pts) | `MultinomialNaiveBayes` **implemented from scratch** — log-space arithmetic, Lidstone smoothing, and a single matrix product instead of a per-document loop — verified to machine precision against `sklearn.naive_bayes.MultinomialNB`. |
| 6a | Part 6a (25 pts) | Cartesian grid search over `vectorizer` x `ngram_range` x `max_features` x `alpha` (32 permutations) wrapped in stratified 5-fold cross-validation, with the vectoriser refitted **inside** each fold and the CV run on our own implementation. All permutations tabulated; the winner reported separately. |
| 4 | Part 4 (5 pts) | Rebuilds the winning configuration and retrains on the full training set. |
| 5 | Part 5 (10 pts) | First 5 test predictions, macro-F1, per-class report, confusion matrix, plus a check of the score on seen vs. genuinely unseen test tweets that explains the gap against the CV estimate. |
| 6c | Part 6c (10 pts) | Explainability: per-class log-likelihood ratios for the most characteristic terms, plus an exact per-term decomposition of individual predictions. |

**Quality metric:** macro-average F1 (multi-class, no central class), as the assignment specifies.

**Result:** macro-F1 = **0.9772** on the held-out test set (5-fold CV estimate on the training
set: 0.9249). The single most influential grid axis was `max_features`: capping the vocabulary
at 5,000 terms costs ~0.19 macro-F1 versus keeping it whole. The notebook also splits the test
score by seen vs. genuinely unseen tweets and explains the gap against the CV estimate.

## Runtime

The whole notebook runs top to bottom in about a minute on a laptop — the grid search
(32 configurations x 5 folds = 160 fits) is the slowest cell at roughly 35-60 seconds.

The notebook is committed **with all outputs stored**, so it can be read and graded
without being re-run.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sklearn'`** — your kernel isn't using the `.venv` environment. Re-run step 4 above, then pick the correct kernel in the notebook UI and restart it.
- **`FileNotFoundError` for the CSVs** — make sure you're running Jupyter from inside the `ml_course_twitter_analysis` folder (or opening the notebook from there), so relative paths resolve correctly.