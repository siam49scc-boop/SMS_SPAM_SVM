# SMS Spam Classification using SVM

An educational machine-learning project that classifies an English SMS as
**Spam** or **Ham** (a normal message). It combines TF-IDF text features with
six features extracted from the original message, then trains a Linear SVM.

See [NOTICE.md](NOTICE.md) for source credit and redistribution information.

## What is included

- A runnable Jupyter/Colab notebook with explanations, EDA and word clouds.
- Training and prediction scripts that accept raw SMS messages.
- An attributed UCI dataset, a trained model and measured evaluation results.
- Confusion matrix, precision–recall curve, top features and error analysis.
- Tests for raw feature extraction, data validation and model serialization.

## How the model works

1. Validate labels and remove repeated messages before splitting the data.
2. Make an 80% training / 20% test split, stratified by label, with seed 42.
3. Lowercase text; remove URLs, email addresses, digits and punctuation;
   remove English stopwords; apply WordNet noun lemmatization.
4. Extract six signals from the **raw SMS**: exclamations, digits, uppercase
   ratio, message length, URL presence and money/bait-word presence.
5. Combine TF-IDF text vectors and scaled numeric signals inside one pipeline.
6. Tune TF-IDF settings, SVM C and class weight using five training folds and
   **spam F1**. The default SVM is balanced; the full search also tries no weighting.
7. Evaluate the selected model on the held-out test set; save charts, metrics,
   the fitted pipeline and misclassified messages.

TF-IDF vocabulary, IDF values and numeric scaling are learned inside each
training fold. The test set is excluded from hyperparameter selection. The best
cross-validation score is a **selection score**, not a separate unbiased estimate.

## Dataset

The included [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
copy contains **5,574 messages**: 4,827 ham and 747 spam. Some Kaggle copies
contain 5,572 messages. Row counts after cleaning are recorded in
[`results/metrics.json`](results/metrics.json).

Creators: Tiago Almeida and Jos Hidalgo. Dataset license: CC BY 4.0.
See [data/README.md](data/README.md) for the citation and conversion details.

## Quick start on Windows

Use **Python 3.11 or 3.12** for a straightforward setup. Open a terminal inside
this project folder, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py "Congratulations! You won a free prize. Call 12345 now!"
.\.venv\Scripts\python.exe predict.py "Hi, are we meeting tomorrow at 3 PM?"
```

If your installed Python is 3.11, use `py -3.11` in the first command. You do not
need to activate the virtual environment or change PowerShell execution policy.
The included model can be used immediately after installation. Internet access
is required for package installation and the first NLTK resource download.

To train your own copy:

```powershell
.\.venv\Scripts\python.exe train.py
```

The full search evaluates 24 configurations across five folds (120 fits).
On a slower laptop, start with `train.py --quick` (two configurations / 10 fits).
The default uses one CPU worker to reduce memory use. Training overwrites the
model and results; the README's included-run table is a snapshot of the original run.

To open the notebook locally:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `SMS_Spam_SVM.ipynb`, then select **Run → Run All Cells**.

## Google Colab: no local installation

1. Open https://colab.research.google.com/ and upload `SMS_Spam_SVM.ipynb`.
2. Run the notebook's first code cell.
3. When prompted, upload the complete project ZIP you downloaded.
4. Choose **Runtime → Run all**. The notebook installs dependencies, loads the
   included dataset, trains the model and displays results.

The notebook uses the full grid by default. Set `QUICK_MODE = True` in the
configuration cell for a shorter demonstration. Colab files are temporary;
the final optional cell downloads the generated model and results as a ZIP.

## macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/python predict.py "Free prize! Call now!"
```

## Python API

Run from the project folder:

```python
import joblib
from sms_classifier import setup_nltk, predict_sms

setup_nltk()
model = joblib.load("models/sms_spam_model.joblib")
print(predict_sms(model, "Hi, see you tomorrow."))
print(model.predict(["Free prize! Call now!", "Lunch at 1 PM?"]))
# Numeric predictions: 0 = ham, 1 = spam.
```

The model includes the preprocessing pipeline. Keep `sms_classifier.py` in
the project and use the pinned scikit-learn version, or retrain. It outputs class
labels and decision scores, not calibrated probability/confidence percentages.
Precision–recall curves can use decision scores directly; calibration is not
required to draw them.

## Files

| File / folder | Purpose |
| --- | --- |
| `SMS_Spam_SVM.ipynb` | Explained notebook with saved example outputs |
| `sms_classifier.py` | Cleaning, feature extraction, data loading and pipeline |
| `train.py` | Hyperparameter search, evaluation and model saving |
| `predict.py` | Single-message and interactive terminal prediction |
| `requirements.txt` | Dependencies; scikit-learn pinned to the tested version |
| `data/` | Dataset, source attribution and original UCI readme |
| `models/` | Trained pipeline and compatibility information |
| `results/` | Measured metrics, charts, features and misclassified messages |
| `tests/` | Focused checks for data and model reuse |
| `HOW_TO_UPLOAD.md` | Beginner-friendly GitHub upload instructions |
| `PROJECT_EXPLANATION_BN.md` | Short Bangla explanation for presentation/viva |
| `NOTICE.md` | Credit for the original project |

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Limitations and next experiments

This is an educational English SMS classifier, not a production spam filter.
The dataset is old and mostly reflects older English SMS patterns. It has not
been evaluated on Bangla, transliterated Bangla or modern messaging-app scams.
False positives and missed spam remain possible. Inspect the error-analysis
CSV and avoid repeatedly tuning against the held-out test set.

Useful next experiments: character n-grams for obfuscated words, a comparison
with ComplementNB / LogisticRegression on the same split, or a small web demo.

## Included run: measured results

Full 24-configuration search; 4,128 training messages and 1,032 held-out test messages.
Duplicate removal leaves 5,160 usable messages.

| Metric | Result |
| --- | ---: |
| Test accuracy | 99.32% |
| Spam precision | 98.40% |
| Spam recall | 96.09% |
| Spam F1 | 97.23% |
| Average precision | 98.99% |
| Always-ham baseline accuracy | 87.60% |
| Best training CV F1 (selection score) | 94.26% |

The test set contains 904 ham and 128 spam: 902 ham and 123 spam were
classified correctly, with **2 false positives and 5 false negatives**.
These are results on this particular dataset/split, not a real-world guarantee.
The original README’s iPhone-prize example is also missed by the included model;
the notebook includes it as a challenging example rather than claiming perfect detection.

![Confusion matrix](results/confusion_matrix.png)

![Precision-recall curve](results/precision_recall_curve.png)

## References

- Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset].
  https://doi.org/10.24432/C5CC84.
- [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Precision–recall from decision scores](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [Original project](https://github.com/Fuwad9096/svm_AI_project)
