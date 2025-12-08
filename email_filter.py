import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# LOAD + TRAIN MODEL

file_path = "spam_dataset.csv"
file = pd.read_csv(file_path)

file['message_content'] = file['message_content'].astype(str).fillna("")
file['label'] = file['is_spam']

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=200,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(file['message_content'])
y = file['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

model.fit(X_train, y_train)

# ROUTES

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")




@app.route('/api', methods=['POST'])
def api_predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    email = data.get("email", "")
    print(email)
    name = data.get("name", "User")

    if email.strip() == "":
        return jsonify({"error": "Email content missing"}), 400

    # Convert email to TF-IDF
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
