import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer


import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# path to dataset file
file_path = "spam_dataset.csv"

file = pd.read_csv(file_path)#, #encoding="latin1")

#file['message_content'] = file['message_content'].astype(str).fillna("")
# Convert labels to binary
file['label'] = file['is_spam']

# vectorizer = HashingVectorizer(
#     lowercase=True, #converts everything to lowercase
#     stop_words='english',
#     n_features=5000,  # number of columns
#     alternate_sign=False
#     #ngram_range=(1, 2)
# )

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=200, 
    ngram_range=(1, 2)
)


X = vectorizer.fit_transform(file['message_content'])
y = file['label']

# # Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Train XGBoost
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42,
    tree_method="hist",
    predictor="gpu_predictor"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
#print("pred", y_pred)

def predict_email(model, vectorizer, email, name):
    # Transform the email using the trained vectorizer
    email_tfifile = vectorizer.transform([email])
    
    # Predict (0 = ham, 1 = spam)
    pred = model.predict(email_tfifile)[0]
    prob = model.predict_proba(email_tfifile)[0][1]
    #print(prob)

    #model.predict_proba()
    
    label = "SPAM" if prob >= 0.90 else "HAM"

    print(f"{name}: {label} (prob = {prob})")
    return label, prob

email_1 = "Congratulations! You've won a $500 Amazon gift card. Click here to claim your prize!"
email_2 = "Hey, can you send me the report from yesterday's meeting?"
email_3 = "Claim your free gift now! Limited time offer. This special offer is available for a limited time only. Act quickly to secure your spot. For more details, visit our website or contact us directly."
email_4 = "Dear Barbara, Thank you for reaching out. I have attached the requested document. Please review and let me know if you have any questions. Audience expect eye address. Reality control not clearly including where through that. Last audience western probably. Yourself memory should notice. Happen item science how ability data. Sincerely, Daniel Boyd Your inputis valuable. Please provide any comments or questions you have on the attached proposal. If you have any questions, please feel free to reach out."
email_5 = "This is a legit email"
email_6 = "Subject: imbalance gas just in case worse comes to worse . - - - - - - - - - - - - - - - - - - - - - - forwarded by mary poorman / na / enron on 03 / 21 / 2001 11 : 29 am - - - - - - - - - - - - - - - - - - - - - - - - - - - from : juliann kemp / enron @ enronxgate on 03 / 21 / 2001 10 : 39 am to : mary poorman / na / enron @ enron cc : subject : imbalance gas mary we just have two . thanks - julie contract 012 - 87794 - 02 - 001 ( delivery ) meter 981506 we owe them 21 , 771 981244 ( delivery ) koch refinery we owe them 16 , 810"
email_7 = "Hello, kindly give me your social security number and your mothers maiden name."
email_8 = "Hey Jim, give me a shout when you have a minute. Thanks, Mike"
email_9 = "Hi son, this is gramma, Please give monies."
email_10 = "abcdefghijklmnopqrstuvwxyz now i know my abcs."
email_11 = "I have hacked your account. Mwahahahah"
email_12 = "good morning everyone cheryl told me that Mike is working from home this morning because he his kid has a doctor's appointment. He will return to office hopefully around 10 tomorrow morning. Please feel free to call Cheryl if you need assistance. Thanks and have a great day, keegan farrell"
email_13 = "Ok, Iknow this is blatantly OT but I'm beginning to go insane. Had an old Dell Dimension XPS sitting in the corner and decided to put it to use, I know it was working pre being stuck in the corner, but when I plugged it in, hit the power nothing happened."



predict_email(model, vectorizer, email_1, "email_1")
predict_email(model, vectorizer, email_2, "email_2")
predict_email(model, vectorizer, email_3, "email_3")
predict_email(model, vectorizer, email_4, "email_4")
predict_email(model, vectorizer, email_5, "email_5")
predict_email(model, vectorizer, email_6, "email_6")
predict_email(model, vectorizer, email_7, "email_7")
predict_email(model, vectorizer, email_8, "email_8")
predict_email(model, vectorizer, email_9, "email_9")
predict_email(model, vectorizer, email_10, "email_10")
predict_email(model, vectorizer, email_11, "email_11")
predict_email(model, vectorizer, email_12, "email_12")
predict_email(model, vectorizer, email_13, "email_13")