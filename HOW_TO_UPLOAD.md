# Put this project on your GitHub: step by step

The project is prepared locally. These steps publish it from **your account**.

## 1. Extract the downloaded ZIP

1. Right-click `SMS_Spam_SVM_GitHub_Ready.zip` in Windows.
2. Select **Extract All**.
3. Open the extracted `sms-spam-svm` folder.
4. You should see `README.md`, `SMS_Spam_SVM.ipynb`, `train.py`, `predict.py`,
   and folders named `data`, `models`, `results`, and `tests`.

Upload the contents of this folder, not the ZIP itself. The project README
must sit directly in the root of your GitHub repository.

## 2. Create a repository

1. Sign in at https://github.com/.
2. Open https://github.com/new.
3. Select **your own account** as the Owner.
4. Repository name: `sms-spam-classification-svm`
5. Description:

   `SMS spam classification with TF-IDF, raw-message features and a Linear SVM, including evaluation and a runnable notebook.`

6. Choose **Private** while preparing. Choose **Public** when you have your
   friend's permission to redistribute the adapted source.
7. Leave **Add a README**, **Add .gitignore**, and **Choose a license** unchecked:
   the first two are already included, and no original code license was supplied.
8. Click **Create repository**.

## 3. Upload the project

1. On the empty repository page, click **uploading an existing file**.
   For a non-empty repository, use **Add file → Upload files**.
2. Open the extracted `sms-spam-svm` folder in File Explorer.
3. Select its **contents** (Ctrl+A), including all four subfolders.
4. Drag the selection into the upload area in your browser. Dragging folders
   preserves their paths; choosing individual files inside a folder may flatten them.
5. Confirm paths such as `data/spam.csv`, `models/sms_spam_model.joblib`, and
   `results/confusion_matrix.png` appear. The top-level README should be
   `README.md`, not `sms-spam-svm/README.md`.
6. Commit message: `Add SMS spam SVM project with notebook and evaluation`
7. Commit to `main` and click **Commit changes** (or the corresponding commit button).

If dragging folders is unsupported in your browser, use GitHub Desktop or the
optional Git commands below; do not upload every nested file to the root.

## 4. Check the result

1. Open your repository's **Code** page. The formatted README should appear.
2. Open `SMS_Spam_SVM.ipynb`; GitHub should render its cells and saved outputs.
3. Open `results/confusion_matrix.png` and `results/precision_recall_curve.png`.
4. Keep the original project credit in `README.md` and `NOTICE.md`.
5. Open **About → gear icon**, add the description, and optionally add topics:
   `machine-learning`, `svm`, `nlp`, `spam-detection`, `python`, `scikit-learn`.
6. Your repository address will be:
   `https://github.com/YOUR_USERNAME/sms-spam-classification-svm`

GitHub displays files and notebook outputs; it does not run Python training.
Use Colab or your computer to run the code.

## 5. Easiest way to run: Colab

1. Open https://colab.research.google.com/.
2. Select **File → Upload notebook** and choose `SMS_Spam_SVM.ipynb`.
3. Run the first code cell, then upload `SMS_Spam_SVM_GitHub_Ready.zip` when prompted.
4. Select **Runtime → Run all**.
5. Use the last prediction cells to change the sample SMS messages.

Alternatively, open the notebook through Colab's **GitHub** tab after publishing
your repository, then still upload the project ZIP when the setup cell asks.

## 6. Run on Windows instead

Python 3.11 or 3.12 is recommended for this dependency set. Check installed
versions with `py -0p`. If necessary, install Python from https://www.python.org/downloads/.

Inside the project folder, right-click → **Open in Terminal**, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py "Congratulations! You won a free prize. Call now!"
.\.venv\Scripts\python.exe predict.py "Hi, see you tomorrow."
.\.venv\Scripts\python.exe train.py --quick
```

Use `py -3.11` if that is your installed version. For the full experiment, run
`train.py` without `--quick`. For an interactive prediction demo, run
`predict.py` without a message. NLTK downloads stopwords and WordNet on first use.

## Optional: Git command-line upload

Use this alternative only if Git is installed and you prefer terminal uploads.
Create an empty repository first. Replace `YOUR_USERNAME` in the remote URL:

```bash
git init
git add .
git commit -m "Add SMS spam SVM project with notebook and evaluation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sms-spam-classification-svm.git
git push -u origin main
```

Run these commands **inside** the extracted `sms-spam-svm` folder. Git may ask
you to sign in or configure your name/email. Do not type your GitHub account
password as an HTTPS Git password; use the browser sign-in offered by Git's
credential manager. If the remote already contains commits, use the browser
upload method or clone the remote before adding files.

## Simple troubleshooting

| Problem | Action |
| --- | --- |
| GitHub contains only a ZIP | Extract it locally and upload the extracted contents. |
| README does not appear at the bottom of Code | Make sure `README.md` is at the repository root. |
| `No module named sms_classifier` | Keep `sms_classifier.py` and run from the project folder. |
| Model missing | Run `python train.py` using the virtual-environment Python. |
| NLTK download fails | Check internet access; run the NLTK downloader mentioned in the error. In a managed network, ask the administrator to provide the two corpora. |
| Model/version compatibility warning | Install the supplied requirements or retrain with your chosen version. |
| Training is slow | Use `--quick`, or use Colab. Keep `--jobs 1` on low-memory laptops. |
| Notebook unavailable in Colab | Upload the `.ipynb`, then upload the complete project ZIP in the first cell. |
