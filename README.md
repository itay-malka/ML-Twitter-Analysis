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

1. **Load & explore** the training/validation CSVs (class distribution, missing values, duplicates).
2. **Clean & vectorize** tweet text (lowercasing, URL/mention removal, `CountVectorizer`).
3. **Train** a Multinomial Naive Bayes classifier.
4. **Evaluate** on the validation set using accuracy and macro-average F1, plus a confusion matrix.
5. Includes a **grid search + 5-fold cross-validation skeleton** for tuning `alpha` and vectorizer settings.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sklearn'`** — your kernel isn't using the `.venv` environment. Re-run step 4 above, then pick the correct kernel in the notebook UI and restart it.
- **`FileNotFoundError` for the CSVs** — make sure you're running Jupyter from inside the `ml_course_twitter_analysis` folder (or opening the notebook from there), so relative paths resolve correctly.