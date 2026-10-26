# Credits and source history

This is an adaptation of the SMS Spam SVM project .
The supplied archive included `README.md` and `sms_spam_svm_upgraded.py`.
Its preprocessing approach, six hand-crafted features and SVM search design
informed this implementation.

This prepared version adds an executable notebook, reusable raw-input pipeline,
separate training and prediction commands, validated dataset loading, an included
attributed UCI dataset, saved results and tests. Model selection uses training
folds only; final results come from a held-out test set. Precision–recall uses
decision scores directly.

No source-code license was included in the supplied original archive. This
adaptation does not invent or assign a license to the friend's code. Obtain the
original author's permission before public redistribution, or use GitHub's Fork
workflow for an unchanged copy. Keep this credit when sharing the adaptation.

The dataset's CC BY 4.0 terms are documented separately in `data/README.md`.
