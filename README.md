# 🎬 Movie Sentiment Analysis with Simple RNN

A complete sentiment analysis system using a **Simple Recurrent Neural Network (RNN)** trained on IMDB movie reviews. This project includes model training, deployment via Streamlit, and word embedding visualization.

---

## Features

- **Simple RNN architecture** with Embedding layer
- **Streamlit web app** for real-time sentiment prediction
- **Early stopping** during training for optimal model performance
- **Word embedding visualization** for text preprocessing
- **Model checkpointing** with .h5 file format

---

## 📂 Repository Structure

| File | Purpose |
|------|---------|
| `main.py` | Streamlit app for sentiment analysis |
| `simple_rnn_imdb.h5` | Trained RNN model |
| `embedding.ipynb` | Word embedding visualization |
| `simplernn.ipynb` | Model training and evaluation |
| `prediction.ipynb` | Prediction examples and analysis |
| `requirements.txt` | Python dependencies |

---

## 🛠️ Technical Implementation

### Model Architecture

model = Sequential([
Embedding(max_features, 128, input_length=max_len),
SimpleRNN(128, activation='relu'),
Dense(1, activation='sigmoid')
])


### Training Details
- **Optimizer:** Adam
- **Loss:** Binary crossentropy
- **Early stopping:** Monitors validation loss with patience=5
- **Validation split:** 20%
- **Final Accuracy:** ~94% (training), ~81% (validation)

### Deployment

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
user_input = st.text_area('Movie Review')
if st.button('Classify'):
processed = preprocess_text(user_input)
prediction = model.predict(processed)
sentiment = 'Positive' if prediction > 0.5 else 'Negative'


---

##  Getting Started

### Installation

pip install -r requirements.txt


### Usage
1. **Train the model** (via `simplernn.ipynb`)
2. **Run the Streamlit app**:

streamlit run main.py

3. **Enter a movie review** in the web interface to get sentiment prediction

---

##  Workflow

1. **Data Preprocessing**
   - Tokenize text using IMDB word index
   - Pad sequences to fixed length (500 words)
     
padded_review = sequence.pad_sequences([encoded_review], maxlen=500)


2. **Model Training**
- 10 epochs with early stopping
- Batch size: 32
- Validation split: 0.2

3. **Deployment**
- Real-time predictions via Streamlit
- User-friendly web interface

---

## 📊 Results

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 94% | 81% |
| **Loss** | 0.126 | 0.544 |

**Sample Prediction:**

Input: "This movie was absolutely fantastic"

Output: Positive (0.98 confidence)

