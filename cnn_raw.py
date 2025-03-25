import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.models import load_model

# Load your CSV data
df = pd.read_csv('bluesky_news_timeline_classified.csv')  

# Preprocess the data
texts = df['text'].tolist()
categories = df['category'].tolist()

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(categories)

# Split data into training and testing sets
train_texts, test_texts, train_categories, test_categories = train_test_split(
    texts, categories, test_size=0.2, random_state=42)

train_labels = label_encoder.transform(train_categories)
test_labels = label_encoder.transform(test_categories)

# Tokenize the text
#tokenizer = Tokenizer(num_words=5000)  # You can adjust num_words
tokenizer = Tokenizer(num_words=5000, char_level=True) # Raw
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to have the same length
max_sequence_length = 100  # You can adjust this based on your data
train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Convert categories to numerical labels (if necessary)
# You might need to use LabelEncoder from scikit-learn
# ...

# Build the CNN model
model = Sequential()
model.add(Embedding(5000, 128))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))  # Adjust activation based on your problem
model.add(Dense(len(set(encoded_labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # Adjust loss and metrics based on your problem

# Train the model
model.fit(train_padded, np.array(train_labels), epochs=10, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(test_padded, np.array(test_labels))
print('Test accuracy:', accuracy)

joblib.dump(tokenizer, "cnn_tokenizer_raw.pkl")
joblib.dump(label_encoder, "cnn_label_encoder_raw.pkl")
model.save("cnn_text_model_raw.h5")

def predict_category(text, model_path="cnn_text_model.h5", tokenizer_path="cnn_tokenizer.pkl", label_encoder_path="cnn_label_encoder.pkl", max_len=100):
  """
  从文件加载模型和辅助文件，对文本进行分类预测
  参数:
      - text: 要分类的文本字符串
      - model_path: CNN 模型文件路径
      - tokenizer_path: Tokenizer 保存路径 (.pkl)
      - label_encoder_path: LabelEncoder 保存路径 (.pkl)
      - max_len: 文本序列填充长度（与训练时一致）
  返回:
      - 预测的分类标签
  """
  if not text or not isinstance(text, str):
      return "Invalid input"

  # 加载模型和预处理器
  model = load_model(model_path)
  tokenizer = joblib.load(tokenizer_path)
  label_encoder = joblib.load(label_encoder_path)

  # 文本处理
  sequence = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(sequence, maxlen=max_len)

  # 预测
  prediction = model.predict(padded)
  predicted_index = prediction.argmax(axis=1)[0]
  predicted_label = label_encoder.inverse_transform([predicted_index])[0]

  return predicted_label

# Make predictions
new_post = "Solar panels become less effective the farther spacecraft travel from the Sun. For missions to the outer solar system and beyond, NASA uses RTGs - nuclear-powered batteries that can provide consistent electricity for decades. An astrophysicist explains: buff.ly/K6gBcyP"
predictions = predict_category(new_post, model_path="cnn_text_model_raw.h5", tokenizer_path="cnn_tokenizer_raw.pkl", label_encoder_path="cnn_label_encoder_raw.pkl")
print('Predicted category:', predictions)

