# Medical Note Classification Project

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "text": [
        "Patient has high fever",
        "Fever and chills",
        "Body pain and fever",
        "High temperature and weakness",
        "Blood sugar is high",
        "Frequent urination and thirst",
        "High glucose level",
        "Diabetes symptoms observed",
        "Chest pain and breathlessness",
        "Heart discomfort",
        "Shortness of breath",
        "Heart disease symptoms",
        "Cough and cold",
        "Throat infection",
        "Viral infection symptoms",
        "Sneezing and cough"
    ],
    "label": [
        "Fever","Fever","Fever","Fever",
        "Diabetes","Diabetes","Diabetes","Diabetes",
        "Heart Disease","Heart Disease","Heart Disease","Heart Disease",
        "Infection","Infection","Infection","Infection"
    ]
}

df = pd.DataFrame(data)

# Train on full data
X = df["text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Accuracy (training)
y_pred = model.predict(X_vec)
print("Model Accuracy:", accuracy_score(y, y_pred))

# Prediction
while True:
    user_input = input("\nEnter medical note (type 'exit' to stop): ")
    
    if user_input.lower() == "exit":
        break
    
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    
    print("Predicted Category:", prediction)