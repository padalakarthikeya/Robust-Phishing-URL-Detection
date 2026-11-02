import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "data/phishing_site_urls.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.sample(
    n=100000,
    random_state=42
)

print("Dataset loaded successfully!")
print("\nDataset shape:")
print(df.shape)

df = df.rename(columns={
    "URL": "url",
    "Label": "label"
})

df["label"] = df["label"].map({
    "good": 0,
    "bad": 1
})

df = df.dropna(
    subset=["url", "label"]
)

df["url"] = df["url"].astype(str).str.lower().str.strip()

df = df.drop_duplicates(
    subset=["url"]
)

print("\nAfter preprocessing:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())


def extract_domain(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        return urlparse(url).netloc.lower()

    except:
        return ""


df["domain"] = df["url"].apply(extract_domain)

print("\nExample data:")
print(df[["url", "domain", "label"]].head())


domains = df["domain"].unique()

np.random.seed(42)
np.random.shuffle(domains)

split_index = int(len(domains) * 0.8)

train_domains = set(domains[:split_index])
test_domains = set(domains[split_index:])

train_df = df[
    df["domain"].isin(train_domains)
].copy()

test_df = df[
    df["domain"].isin(test_domains)
].copy()

print("\nTraining URLs:", len(train_df))
print("Testing URLs:", len(test_df))
print("Training domains:", len(train_domains))
print("Testing domains:", len(test_domains))


vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["url"]
)

X_test = vectorizer.transform(
    test_df["url"]
)

y_train = train_df["label"]
y_test = test_df["label"]

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n")
print("=" * 60)
print("CROSS-DOMAIN PHISHING URL DETECTION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Legitimate",
        "Phishing"
    ],
    yticklabels=[
        "Legitimate",
        "Phishing"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Cross-Domain Phishing URL Detection")
plt.tight_layout()
plt.show()


def predict_url(url):

    url = str(url).lower().strip()

    features = vectorizer.transform(
        [url]
    )

    prediction = model.predict(
        features
    )[0]

    probability = model.predict_proba(
        features
    )[0][1]

    print("\n")
    print("=" * 60)

    print("URL:", url)

    if prediction == 1:
        print("Prediction: PHISHING")
    else:
        print("Prediction: LEGITIMATE")

    print(
        f"Phishing probability: {probability:.2%}"
    )

    print("=" * 60)


predict_url(
    "https://www.google.com"
)

predict_url(
    "https://www.microsoft.com"
)

predict_url(
    "http://paypal-login-security-verify.xyz/login"
)

predict_url(
    "http://secure-account-verification-login.xyz/update"
)