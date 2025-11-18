import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import HashingVectorizer


import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# path to dataset file
file_path = "C:\\Users\\Travi\\Documents\\368Project\\spam_dataset.csv"

file = pd.read_csv(file_path)
#file['message_content'] = file['message_content'].astype(str).fillna("")
# Convert labels to binary
file['label'] = file['is_spam']

vectorizer = HashingVectorizer(
    lowercase=True, #converts everything to lowercase
    stop_words='english',
    n_features=2,  # number of columns
    #ngram_range=(1, 2)
)

X = vectorizer.transform(file['message_content'])
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
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
#print("pred", y_pred)

def predict_email(model, vectorizer, email):
    # Transform the email using the trained vectorizer
    email_tfifile = vectorizer.transform([email])
    
    # Predict (0 = ham, 1 = spam)
    pred = model.predict(email_tfifile)[0]
    prob = model.predict_proba(email_tfifile)[0][1]
    #print(prob)

    #model.predict_proba()
    
    # Display the result
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"pred: {label} (prob = {prob})")
    return label, prob

sample_email_1 = "Congratulations! You've won a $500 Amazon gift card. Click here to claim your prize!"
sample_email_2 = "Hey, can you send me the report from yesterday's meeting?"
sample_email_3 = "Claim your free gift now! Limited time offer. This special offer is available for a limited time only. Act quickly to secure your spot. For more details, visit our website or contact us directly."
sample_email_4 = "Dear Barbara, Thank you for reaching out. I have attached the requested document. Please review and let me know if you have any questions. Audience expect eye address. Reality control not clearly including where through that. Last audience western probably. Yourself memory should notice. Happen item science how ability data. Sincerely, Daniel Boyd Your inputis valuable. Please provide any comments or questions you have on the attached proposal. If you have any questions, please feel free to reach out."
sample_email_5 = "This is a legit email"
sample_email_6 = "Subject: imbalance gas just in case worse comes to worse . - - - - - - - - - - - - - - - - - - - - - - forwarded by mary poorman / na / enron on 03 / 21 / 2001 11 : 29 am - - - - - - - - - - - - - - - - - - - - - - - - - - - from : juliann kemp / enron @ enronxgate on 03 / 21 / 2001 10 : 39 am to : mary poorman / na / enron @ enron cc : subject : imbalance gas mary we just have two . thanks - julie contract 012 - 87794 - 02 - 001 ( delivery ) meter 981506 we owe them 21 , 771 981244 ( delivery ) koch refinery we owe them 16 , 810"

predict_email(model, vectorizer, sample_email_1)
predict_email(model, vectorizer, sample_email_2)
predict_email(model, vectorizer, sample_email_3)
predict_email(model, vectorizer, sample_email_4)
predict_email(model, vectorizer, sample_email_5)
predict_email(model, vectorizer, sample_email_6)