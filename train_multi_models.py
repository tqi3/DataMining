import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics

def train_model(text, category, model_name, model, model_file_path):
    """ train tfidf + model """
    X_train, X_test, y_train, y_test = train_test_split(
        text, category, test_size=0.2, random_state=42
    )

    # train TF-IDF + Model
    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        (model_name, model)
    ])

    model_pipeline.fit(X_train, y_train)

    # evaluate the model, test the accuracy
    y_pred = model_pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"model [{model_name}] train finished, accuracy: {accuracy:.4f}")

    # save the model
    joblib.dump(model_pipeline, model_file_path)
    print(f"Model saved: {model_file_path}")

# define the cleaning method
def clean_text(text):
    text = text.lower()  # transfer to lower case
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove website links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep letters only [a-zA-Z]
    return text.strip()

# predict new posts
def predict_category(text, model_filepath, clean):
    if clean:
        text = clean_text(text)
    model = joblib.load(model_filepath)  # load previous trained
    return model.predict([text])[0]
    
# load the database from .csv file
df = pd.read_csv('bluesky_news_timeline_classified.csv')
# Apply to clean text
df["clean_text"] = df["text"].apply(clean_text)

# raw and also cleaned (processed)
datasets = [("raw", df["text"]), ("processed", df["clean_text"])]

# define models
models = [
    ("LinearSVC(svm)", LinearSVC()),
    ("MultinomialNB(naive_bayes)", MultinomialNB()),
    ("LogisticRegression", LogisticRegression()),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)),
]

new_post = "Solar panels become less effective the farther spacecraft travel from the Sun. For missions to the outer solar system and beyond, NASA uses RTGs - nuclear-powered batteries that can provide consistent electricity for decades. An astrophysicist explains: buff.ly/K6gBcyP"
print(f"\nTest post： {new_post}\n")
for name, dateset in datasets:
    print(f"{name}: \n")
    for model_name, model in models:
        model_file_path = name + "_" + model_name + ".pkl"
        train_model(dateset, df["category"], model_name, model, model_file_path)
        # test the new post
        predicted = predict_category(new_post, model_file_path, False)
        print(f"{model_name} predict: {predicted}\n\n")
