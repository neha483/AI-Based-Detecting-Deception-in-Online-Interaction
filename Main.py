from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import string

from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN

# Tkinter UI
main = Tk()
main.title("Dishonest Internet Users Detection")
main.geometry("1300x1200")

global df, model
global texts, labels, preprocessed_texts, X_tfidf_dense
global X_train, X_val, y_train, y_val, model, dt
global tfidf_vectorizer


def uploadDataset():
    global df, texts, labels
    file_path = filedialog.askopenfilename(
        title="Select Dataset File", filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        df = pd.read_csv(file_path)
        text.insert(END, "Dataset loaded\n")
        text.insert(END,str(df))
        texts = df["text"].tolist()
        labels = df["Label"].tolist()


def preprocessText(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a single string
    cleaned_text = " ".join(tokens)

    return cleaned_text


def extractTfidfFeatures(texts):
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000
    )  # Adjust max_features as needed

    # Fit the vectorizer and transform the documents
    X_tfidf = tfidf_vectorizer.fit_transform(texts)

    # Convert the TF-IDF matrix to a dense array for better readability
    X_tfidf_dense = X_tfidf.toarray()

    return X_tfidf_dense, tfidf_vectorizer


def splitData(X_tfidf_dense, labels):
    # Convert y_train to numpy array
    y_train = np.array(labels)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_tfidf_dense, labels, test_size=0.1
    )

    return X_train, X_val, y_train, y_val


def preprocessAndSplit():
    global texts, labels, preprocessed_texts
    text.delete("1.0", END)

    # Preprocess the texts
    preprocessed_texts = [preprocessText(text) for text in texts]
    text.insert(END, "Internet Text before preprocessing (first 5 rows):\n")
    for original_text in texts[:5]:
        text.insert(END, str(original_text) + "\n")
    text.insert(END, "\nInternet Text after preprocessing (first 5 rows):\n")
    for preprocessed_text in preprocessed_texts[:5]:
        text.insert(END, str(preprocessed_text) + "\n")


def TF_IDF():
    global texts, labels, preprocessed_texts, X_tfidf_dense, tfidf_vectorizer
    text.delete("1.0", END)

    X_tfidf_dense, tfidf_vectorizer = extractTfidfFeatures(preprocessed_texts)

    text.insert(END, "TF-IDF feature extraction:" + str(X_tfidf_dense) + "\n")
    text.insert(END, "TF-IDF feature Size: " + str(len(X_tfidf_dense)) + "\n")
    text.insert(END, "TF-IDF feature Size: " + str(X_tfidf_dense.shape) + "\n")


def Data_split():
    text.delete("1.0", END)
    global X_train, X_val, y_train, y_val, model, dt

    # Split the data
    X_train, X_val, y_train, y_val = splitData(X_tfidf_dense, labels)

    text.insert(END, "Text preprocessed and data split\n")
    text.insert(END, "Dataset Size: " + str(X_tfidf_dense.shape) + "\n")
    text.insert(END, "Training Dataset Size: " + str(X_train.shape) + "\n")
    text.insert(END, "Testing Dataset Size: " + str(len(y_train)) + "\n")


def performance_evaluation(model_name, y_true, y_pred, classes):

    accuracy = accuracy_score(y_true, y_pred)*100
    pre = precision_score(y_true, y_pred, average="weighted")*100
    rec = recall_score(y_true, y_pred, average="weighted")*100
    f1s = f1_score(y_true, y_pred, average="weighted")*100
    report = classification_report(y_true, y_pred, target_names=classes)
    text.delete('1.0',END)

    text.insert(END, f"{model_name} Accuracy: {accuracy}\n")
    text.insert(END, f"{model_name} Precision: {pre}\n")
    text.insert(END, f"{model_name} Recall: {rec}\n")
    text.insert(END, f"{model_name} F1-score: {f1s}\n")
    text.insert(END, f"{model_name} Classification report\n{report}\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()


def trainRandomForestModel():
    global X_train, X_val, y_train, y_val, model

    model = RandomForestClassifier(
        n_estimators=1, max_depth=10, max_features="sqrt", random_state=42
    )
    model.fit(X_train, y_train)
    text.insert(END, "Random Forest Model trained\n")
    Y_pred = model.predict(X_val)
    classes = ["dishonest", "honest"]
    performance_evaluation("RFC Model", y_val, Y_pred, classes)


def trainDecisionTreeModel():
    global X_train, y_train, dt
    dt = DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    dt.fit(X_train, y_train)
    text.insert(END, "Decision Tree Model trained\n")
    Y_pred = dt.predict(X_val)
    classes = ["dishonest", "honest"]
    performance_evaluation("DTC Model", y_val, Y_pred, classes)


def rnnmodel():
    global X_train, y_train, X_val, y_val, rnn

    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

    model_path = Path("model/rnn_model.h5")
    if model_path.exists():
        # Load the existing model
        rnn = load_model(model_path)
        print("RNN Model loaded from file")

    else:
        rnn = Sequential()
        rnn.add(SimpleRNN(units=50, activation="relu", input_shape=(1, 1000)))
        rnn.add(Dense(units=1, activation="sigmoid"))
        rnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        rnn.fit(
            X_train_reshaped,
            y_train,
            epochs=5,
            batch_size=32,
            validation_data=(X_val_reshaped, y_val),
        )
        rnn.save(model_path)

    loss, accuracy = rnn.evaluate(X_val_reshaped, y_val)

    text.insert(END, "RNN Model trained\n")
    Y_pred = rnn.predict(X_val_reshaped)

    Y_pred_binary = [1 if pred > 0.5 else 0 for pred in Y_pred]

    classes = ["dishonest", "honest"]
    performance_evaluation("RNN Model", y_val, Y_pred_binary, classes)

    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# def predictText():
#     global model
#     global tfidf_vectorizer
#     text.delete("1.0", END)
#     model = load_model(Path("model/rnn_model.h5"))
#     file_path = filedialog.askopenfilename(
#         title="Select Dataset File", filetypes=[("CSV files", "*.csv")]
#     )
#     df = pd.read_csv(file_path)
#     # Check if the 'text' column exists in the DataFrame
#     if "text" in df.columns:
#         # Preprocess the 'text' column
#         preprocessed_texts = [preprocessText(text) for text in df["text"]]
#     else:
#         text.insert(END, "Error: 'text' column not found in the dataset.\n")
#     tfidf_vectorizer = TfidfVectorizer(
#         max_features=1000
#     )  # Adjust max_features as needed
#     X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_texts)

#     # Convert the TF-IDF matrix to a dense array for better readability
#     X_tfidf_dense = X_tfidf.toarray()

#     A = "Honest internet user"
#     B = "Dishonest Internet user"
#     predict_rf = model.predict(X_tfidf_dense)

#     for index, row in df.iterrows():
#         text.insert(END, f"Input Data: {row['text']}\n")
#         predict_rf = model.predict(X_tfidf_dense[index : index + 1])

#         if predict_rf[0] == 0:
#             text.insert(END, f"Random Forest Prediction: {A}\n\n\n")
#         else:
#             text.insert(END, f"Random Forest Prediction: {B}\n\n\n")

def predictText():
    text.delete("1.0", END)
    model = load_model(Path("model/rnn_model.h5"))
    file_path = filedialog.askopenfilename(title="Select Dataset File",          	filetypes=[("CSV files", "*.csv")])
    df = pd.read_csv(file_path)

    if "text" in df.columns:
        preprocessed_texts = [preprocessText(text) for text in df["text"]]
    else:
        text.insert(END, "Error: 'text' column not found in the dataset.\n")

    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_texts)
    X_tfidf_dense = X_tfidf.toarray()

    A = "Honest Internet user"
    B = "Dishonest Internet user"

    for index, row in df.iterrows():
        text.insert(END, f"Input Data: {row['text']}\n")
        X_test_reshaped = np.reshape(X_tfidf_dense[index], (1, 1, 1000))
        predict_rnn = model.predict(X_test_reshaped)

        if predict_rnn[0][0] <= 0.5:
            text.insert(END, f"RNN Prediction: {A}\n\n\n")
        else:
            text.insert(END, f"RNN Prediction: {B}\n\n\n")

font = ("times", 18, "bold")
title = Label(
    main,
    text="AI Based Detecting Deception in Online Interactions: An Analysis of the Dishonest Internet Users",
    justify=LEFT,
)
title.config(bg="lightblue", fg="darkgreen")
title.config(font=font)
title.config(height=3, width=120)
title.pack()

font1 = ("times", 14, "bold")
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20, y=100)
uploadButton.config(font=font1)

preprocessButton = Button(
    main, text="Preprocess and Split Data", command=preprocessAndSplit
)
preprocessButton.place(x=20, y=150)
preprocessButton.config(font=font1)

preprocessButton = Button(main, text="TF-IDF Feature extraction", command=TF_IDF)
preprocessButton.place(x=20, y=200)
preprocessButton.config(font=font1)

preprocessButton = Button(main, text="Dataset Splitting", command=Data_split)
preprocessButton.place(x=20, y=250)
preprocessButton.config(font=font1)


trainRFButton = Button(
    main, text="Train Random Forest Model", command=trainRandomForestModel
)
trainRFButton.place(x=20, y=300)
trainRFButton.config(font=font1)

trainDTButton = Button(
    main, text="Train Decision Tree Model", command=trainDecisionTreeModel
)
trainDTButton.place(x=20, y=350)
trainDTButton.config(font=font1)

trainRNNButton = Button(main, text="Train RNN Model", command=rnnmodel)
trainRNNButton.place(x=20, y=400)
trainRNNButton.config(font=font1)

predictButton = Button(main, text="Predict Text", command=predictText)
predictButton.place(x=20, y=450)
predictButton.config(font=font1)

text = Text(main, height=30, width=85)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500, y=100)
text.config(font=font1)

main.config()
main.mainloop()
