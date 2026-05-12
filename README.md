#  Suicide Risk Detection System

A multi-model NLP pipeline for detecting suicide risk in Reddit posts, combining classical ML, deep learning, RAG retrieval, and LLM-powered explanations.

---

#  Overview

This project classifies Reddit posts as Suicide Risk or Non-Risk using a layered approach:

1. Text Preprocessing — cleaning, tokenization, lemmatization via NLTK
2. Classical ML— TF-IDF + Logistic Regression baseline
3. Deep Learning — RNN, LSTM, GRU, and BiLSTM models (TensorFlow/Keras)
4. Transformer Fine-tuning — BERT (`bert-base-uncased`) for sequence classification
5. RAG Pipeline — FAISS vector store + Sentence Transformers for similar post retrieval
6. LLM Explanation— GPT-4o-mini generates empathetic, context-aware explanations
7. Gradio UI — Interactive web interface for real-time analysis

---

#  Project Structure

```
.
├── data/                          # Raw and processed data
├── models/
│   ├── simple_rnn_model.h5
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   ├── bilstm_model.h5            # Primary inference model
│   ├── bert_model/                # Fine-tuned BERT checkpoint
│   └── tokenizer.pkl              # Keras tokenizer
├── vector_db/
│   ├── faiss_index.bin            # FAISS vector index
│   └── rag_texts.json             # Indexed post texts
├── notebooks/
├── logistic_model.pkl             # Logistic Regression model
├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
└── README.md
```

---

#  Setup & Installation

# 1. Clone the repository

```bash
git clone <your-repo-url>
cd suicide-risk-detection
```

# 2. Install dependencies

```bash
pip install -r requirements.txt
```

# 3. Download the dataset

The project uses the [Suicide Watch dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) from Kaggle. Either:

- Download manually and place `Suicide_Detection.csv` in `suicide-watch/`
- Or use `opendatasets` (requires Kaggle API credentials):

```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch")
```

# 4. Set your API key

Replace the placeholder in the notebook/script with your OpenAI API key:

```python
client = OpenAI(api_key="your-openai-api-key")
```

> Never commit API keys to version control. Use environment variables or a `.env` file.

---

#  Pipeline Walkthrough

# Step 1 — Data Preprocessing
- Lowercasing, URL/digit/punctuation removal
- Stopword filtering and WordNet lemmatization
- Feature engineering: word count, char count, punctuation count

# Step 2 — TF-IDF + Logistic Regression
- `TfidfVectorizer` with `max_features=5000`, bigrams
- Stratified 80/20 train-test split
- Saved as `logistic_model.pkl` + `tfidf_vectorizer.pkl`

# Step 3 — Deep Learning Models
All models use an `Embedding(5000, 128)` layer + `Dropout(0.5)` + `Dense(1, sigmoid)`:

| Model   | Recurrent Layer         |
|---------|------------------------|
| RNN     | `SimpleRNN(64)`        |
| LSTM    | `LSTM(64)`             |
| GRU     | `GRU(64)`              |
| BiLSTM  | `Bidirectional(LSTM(64))`  used in app |

Trained for 5 epochs, batch size 64, 20% validation split.

# Step 4 — BERT Fine-tuning
- `bert-base-uncased` fine-tuned on 20,000 samples for 1 epoch
- Saved to `models/bert_model/`

# Step 5 — RAG with FAISS
- `all-MiniLM-L6-v2` encodes 10,000 posts into 384-dim embeddings
- Stored in a `faiss.IndexFlatL2` index
- At inference, top-3 similar posts are retrieved for the query

# Step 6 — Gradio App
Run the final cell to launch the UI:

```python
demo.launch(share=True, debug=False)
```

The app provides
- Prediction label and confidence score
- Similar posts retrieved via RAG
- GPT-4o-mini explanation with key signals and support resources

---

  Models Summary

| Model              | Type              | Input        |
|--------------------|-------------------|--------------|
| Logistic Regression| Classical ML      | TF-IDF       |
| RNN / LSTM / GRU   | Deep Learning     | Padded seqs  |
| BiLSTM             | Deep Learning     | Padded seqs  |
| BERT               | Transformer       | Tokenized text|
| GPT-4o-mini        | LLM (API)         | Prompt + RAG |

---




## 📄 License

MIT License. See `LICENSE` for details.
