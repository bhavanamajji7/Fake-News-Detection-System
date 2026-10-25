import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "fake_news_merged_rich.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Original Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SELECT TEXT COLUMN
# ============================================================

text_column = "combined_text"

if text_column not in df.columns:
    print("\nERROR: combined_text column not found!")
    print("Available columns:", df.columns.tolist())
    exit()


# ============================================================
# 3. CREATE TARGET LABEL
# ============================================================

# Prefer label_text if available
if "label_text" in df.columns:

    def convert_label(value):

        value = str(value).lower().strip()

        if value in ["fake", "false", "1"]:
            return 1

        elif value in ["real", "true", "0"]:
            return 0

        else:
            return np.nan

    df["target"] = df["label_text"].apply(convert_label)

else:

    df["target"] = pd.to_numeric(df["label"], errors="coerce")


# Remove rows with invalid labels
df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


# ============================================================
# 4. CHECK LABEL DISTRIBUTION
# ============================================================

print("\nLabel Distribution:")

print(df["target"].value_counts())

print("\nLabel Meaning:")
print("0 = Real News")
print("1 = Fake News")


# ============================================================
# 5. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    import re
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = df[text_column].apply(clean_text)


# ============================================================
# 6. REMOVE EMPTY TEXT
# ============================================================

df = df[df["clean_text"].str.len() > 0]


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(subset=["clean_text"])

after = len(df)

print("\nDuplicate texts removed:", before - after)
print("Final Dataset Shape:", df.shape)


# ============================================================
# 8. SPLIT DATA
# ============================================================

X = df["clean_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 9. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")
print("Number of Features:", X_train_tfidf.shape[1])


# ============================================================
# 10. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC(
            C=1.5,
            class_weight="balanced"
        )
}


# ============================================================
# 11. TRAIN MODELS
# ============================================================

results = []

trained_models = {}

predictions = {}

print("\nTraining Models...\n")

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    trained_models[name] = model
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

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

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])


# ============================================================
# 12. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n==============================================")
print("MODEL PERFORMANCE")
print("==============================================")

print(results_df.to_string(index=False))


# ============================================================
# 13. CLASSIFICATION REPORT - SVM
# ============================================================

print("\n==============================================")
print("SVM CLASSIFICATION REPORT")
print("==============================================")

print(
    classification_report(
        y_test,
        predictions["SVM"],
        target_names=["Real", "Fake"],
        zero_division=0
    )
)


# ============================================================
# 14. GRAPH 1 - MODEL PERFORMANCE
# ============================================================

def model_performance_graph():

    results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ].plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# 15. GRAPH 2 - ACCURACY
# ============================================================

def accuracy_graph():

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.xticks(rotation=0)

    for i, value in enumerate(results_df["Accuracy"]):

        plt.text(
            i,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 16. GRAPH 3 - PRECISION
# ============================================================

def precision_graph():

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["Precision"]
    )

    plt.title("Precision Comparison")
    plt.ylabel("Precision")
    plt.ylim(0, 1)

    plt.xticks(rotation=0)

    for i, value in enumerate(results_df["Precision"]):

        plt.text(
            i,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 17. GRAPH 4 - RECALL
# ============================================================

def recall_graph():

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["Recall"]
    )

    plt.title("Recall Comparison")
    plt.ylabel("Recall")
    plt.ylim(0, 1)

    plt.xticks(rotation=0)

    for i, value in enumerate(results_df["Recall"]):

        plt.text(
            i,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 18. GRAPH 5 - F1 SCORE
# ============================================================

def f1_graph():

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df["F1 Score"]
    )

    plt.title("F1 Score Comparison")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)

    plt.xticks(rotation=0)

    for i, value in enumerate(results_df["F1 Score"]):

        plt.text(
            i,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 19. GRAPH 6 - SVM CONFUSION MATRIX
# ============================================================

def svm_confusion_matrix():

    cm = confusion_matrix(
        y_test,
        predictions["SVM"]
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()


# ============================================================
# 20. GRAPH 7 - REAL VS FAKE DISTRIBUTION
# ============================================================

def distribution_graph():

    counts = df["target"].value_counts()

    labels = ["Real News", "Fake News"]

    values = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values
    )

    plt.title("Real vs Fake News Distribution")
    plt.ylabel("Number of News Articles")

    for i, value in enumerate(values):

        plt.text(
            i,
            value + 50,
            str(value),
            ha="center"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 21. TEST NEW NEWS
# ============================================================

def test_new_news():

    print("\n==============================================")
    print("TEST NEW NEWS")
    print("==============================================")

    print("Type 'exit' to return to menu.")

    while True:

        news = input(
            "\nEnter news article: "
        ).strip()

        if news.lower() == "exit":
            break

        if news == "":
            print("Please enter some news text.")
            continue

        cleaned_news = clean_text(news)

        news_vector = vectorizer.transform(
            [cleaned_news]
        )

        prediction = trained_models["SVM"].predict(
            news_vector
        )[0]

        if prediction == 1:

            print("\nPrediction: FAKE NEWS")

        else:

            print("\nPrediction: REAL NEWS")


# ============================================================
# 22. TEST ACTUAL DATASET NEWS
# ============================================================

def test_dataset_news():

    print("\n==============================================")
    print("TEST NEWS FROM DATASET")
    print("==============================================")

    sample = df.sample(
        1,
        random_state=np.random.randint(10000)
    ).iloc[0]

    news = sample["clean_text"]

    actual = sample["target"]

    news_vector = vectorizer.transform(
        [news]
    )

    prediction = trained_models["SVM"].predict(
        news_vector
    )[0]

    print("\nNews:")
    print(news[:1000])

    print("\nActual Label:")

    if actual == 1:
        print("FAKE NEWS")
    else:
        print("REAL NEWS")

    print("\nModel Prediction:")

    if prediction == 1:
        print("FAKE NEWS")
    else:
        print("REAL NEWS")


# ============================================================
# 23. SHOW ALL GRAPHS
# ============================================================

def show_all_graphs():

    model_performance_graph()

    accuracy_graph()

    precision_graph()

    recall_graph()

    f1_graph()

    svm_confusion_matrix()

    distribution_graph()


# ============================================================
# 24. MENU
# ============================================================

while True:

    print("\n")
    print("==============================================")
    print("       FAKE NEWS DETECTION SYSTEM")
    print("==============================================")

    print("1. Model Performance Graph")
    print("2. Accuracy Graph")
    print("3. Precision Graph")
    print("4. Recall Graph")
    print("5. F1 Score Graph")
    print("6. SVM Confusion Matrix")
    print("7. Real vs Fake Distribution")
    print("8. Show All Graphs")
    print("9. Test New News")
    print("10. Test Dataset News")
    print("0. Exit")

    choice = input(
        "\nEnter your choice: "
    ).strip()


    if choice == "1":

        model_performance_graph()


    elif choice == "2":

        accuracy_graph()


    elif choice == "3":

        precision_graph()


    elif choice == "4":

        recall_graph()


    elif choice == "5":

        f1_graph()


    elif choice == "6":

        svm_confusion_matrix()


    elif choice == "7":

        distribution_graph()


    elif choice == "8":

        show_all_graphs()


    elif choice == "9":

        test_new_news()


    elif choice == "10":

        test_dataset_news()


    elif choice == "0":

        print("\nProgram exited successfully!")
        break


    else:

        print("\nInvalid choice. Please select 0-10.")