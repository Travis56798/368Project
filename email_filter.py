import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# LOAD + TRAIN MODEL

file_path = "spam_dataset.csv"

file = pd.read_csv(file_path) #loads file into variable

file['message_content'] = file['message_content'].astype(str).fillna("") #fills in any blank rows with empty strings

file['label'] = file['is_spam']

vectorizer = TfidfVectorizer(
    lowercase=True,  #converts all words to lowercase
    stop_words='english',  #removes common english words, like 'the'
    max_features=200, #stores the most frequent words seen in the data set
    ngram_range=(1, 2) #extracts single words and pairs of words
)

x = vectorizer.fit_transform(file['message_content']) #turns message_content into numerical values
y = file['label'] #stores the corresponding is_spam value(0 or 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y) #splits data into training and testing groups

model = xgb.XGBClassifier(  #sets up a training tree
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=200, #set equal to max_features
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=1,
    tree_method="hist",  #ensures less memory is used
    predictor="gpu_predictor" #speeds up computation time
)

model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],    #trains the model on our data 
    verbose=False
)
y_prob_test = model.predict_proba(x_test)[:, 1]
y_pred_test = (y_prob_test >= 0.80).astype(int)
evaluation_results = {
    "accuracy": float(accuracy_score(y_test, y_pred_test)),
    "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),}
print("Model evaluation:", evaluation_results)
# ROUTES

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html",metrics=evaluation_results)




@app.route('/api', methods=['POST'])
def api_predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    email = data.get("email", "")
    name = data.get("name", "User")

    if email.strip() == "":
        return jsonify({"error": "Email content missing"}), 400

    max_len = 2000  #avioid the over length influences
    if len(email) > max_len:
        email = email[:max_len]

    email_tfidf = vectorizer.transform([email])

    # Predict
    prob = model.predict_proba(email_tfidf)[0][1]
    print(prob)
    label = "SPAM" if prob >= 0.80 else "HAM"

    return jsonify({
        "name": name,
        "label": label,
        "probability": float(prob)
    })


if __name__ == '__main__':
    app.run(debug=True)

'''email_1 = "Congratulations! You've won a $500 Amazon gift card. Click here to claim your prize!"
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
predict_email(model, vectorizer, email_13, "email_13")'''