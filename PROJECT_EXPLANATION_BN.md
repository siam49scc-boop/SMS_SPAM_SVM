# সহজ বাংলায় প্রজেক্টটি বোঝা

## প্রজেক্টের কাজ

একটি ইংরেজি SMS দিলে মডেল বলবে সেটি **Spam** নাকি **Ham**। Ham মানে স্বাভাবিক
মেসেজ। যেমন পুরস্কার জেতার ভুয়া মেসেজ Spam হতে পারে; বন্ধুর সঙ্গে দেখা করার
মেসেজ Ham হতে পারে। এগুলো উদাহরণ—মডেল সব ক্ষেত্রে সঠিক উত্তর দেবে এমন নয়।

## কাজের ধাপ

1. **Dataset:** আগে থেকেই ham/spam লেবেল দেওয়া SMS ব্যবহার করি। একই মেসেজ
   একাধিকবার থাকলে বাদ দিই, যেন সেটি training ও test—দুই ভাগে না যায়।
2. **Split:** ৮০% দিয়ে শিখাই, ২০% দিয়ে শেষে পরীক্ষা করি। দুই ভাগে spam/ham-এর
   অনুপাত কাছাকাছি রাখার জন্য stratified split ব্যবহার করি।
3. **Cleaning:** ছোট হাতের অক্ষর করি; URL, email, সংখ্যা ও punctuation সরাই;
   stopword বাদ দিই; lemmatization করি।
4. **Raw features:** মূল মেসেজ থেকে `!` কতটি, digit কতটি, বড় হাতের অক্ষরের
   অনুপাত, দৈর্ঘ্য, URL আছে কি না এবং money/bait শব্দ আছে কি না বের করি।
5. **TF-IDF:** কোন শব্দ একটি মেসেজে গুরুত্বপূর্ণ কিন্তু সব মেসেজে খুব সাধারণ
   নয়—তা সংখ্যায় প্রকাশ করি। Bigram হলো পাশাপাশি দুটি শব্দ, যেমন `free prize`।
6. **Scaling:** সংখ্যাগত feature-গুলোর scale কাছাকাছি আনি।
7. **SVM:** feature-গুলোর সাহায্যে ham ও spam আলাদা করার decision boundary শেখে।
8. **GridSearchCV:** training data-র ৫টি fold ব্যবহার করে বিভিন্ন parameter পরীক্ষা
   করি এবং spam F1 অনুযায়ী ভালো configuration বেছে নিই।
9. **Evaluation:** শেষে আলাদা test data দিয়ে ফল যাচাই করি এবং model save করি।

## মূল প্রশ্ন: cleaning করলে spam signal হারাবে না?

TF-IDF-এর জন্য text পরিষ্কার করি। কিন্তু মূল মেসেজের সংখ্যা, `!!!`, ALL CAPS
ইত্যাদি আগে আলাদা feature হিসেবে বের করি। তাই পরিষ্কার text ও raw signal—দুই
ধরনের তথ্যই SVM পায়। প্রতিটি signal একা spam প্রমাণ করে না।

## কেন accuracy যথেষ্ট নয়?

ধরা যাক ১০০টি মেসেজে ৮৭টি ham। সব মেসেজকে ham বললেই ৮৭% accuracy হতে পারে,
কিন্তু একটি spam-ও ধরা পড়বে না। তাই spam-এর precision, recall ও F1 দেখি।

- **Precision:** যেগুলোকে spam বলেছি, তার কত অংশ সত্যিই spam?
- **Recall:** আসল spam-এর কত অংশ ধরতে পেরেছি?
- **F1:** precision ও recall-এর harmonic mean; দুই দিক মিলিয়ে মূল্যায়ন।
- **False positive:** আসল ham-কে spam বলা।
- **False negative:** আসল spam-কে ham বলা।

## গুরুত্বপূর্ণ সতর্কতা

Test data দিয়ে parameter বাছাই করি না। TF-IDF ও scaler প্রতিটি training fold
থেকেই শেখে। Grid search-এর best CV score হলো parameter বাছাইয়ের score; final
test score আলাদা। LinearSVC-এর decision score probability নয়। PR curve আঁকতে
decision score ব্যবহার করা যায়, তাই শুধু curve আঁকার জন্য calibration লাগে না।

## কোন ফাইল কী করে?

- `sms_classifier.py`: preprocessing, feature extraction ও pipeline।
- `train.py`: train, tuning, evaluation এবং model save।
- `predict.py`: নতুন SMS-এর prediction।
- `SMS_Spam_SVM.ipynb`: ধাপে ধাপে code, explanation, graph ও sample output।
- `results/metrics.json`: আসল run-এর সংখ্যাগত ফলাফল।


