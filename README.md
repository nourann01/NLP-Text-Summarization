
# 📰 Abstractive Text Summarization — Comparative NLP Study

A comprehensive comparative study of **5 models** for abstractive text summarization, built as part of an undergraduate Natural Language Processing course. Models are trained and evaluated on a real-world news dataset, with a live interactive Streamlit demo to compare all models side by side.

---

## Models Compared

| Model | Type | Key Feature |
|-------|------|-------------|
| Vanilla RNN | Neural (Seq2Seq) | Baseline — single context vector |
| GRU + Attention | Neural (Seq2Seq) | Gating + attention mechanism |
| LSTM + Attention | Neural (Seq2Seq) | Dual-state memory + attention |
| HMM | Statistical | Frequency-based extractive baseline |
| T5 Transformer | Pre-trained | Fine-tuned T5-small — state of the art |

---


## 📁 Dataset

**[therapara/summary-of-news-articles](https://huggingface.co/datasets/therapara/summary-of-news-articles)** — available on Hugging Face.

- 56,216 real news article & summary pairs
- Pre-split: 45k train / 5.6k validation / 5.6k test
- Average compression ratio: 8.2x (articles ~8x longer than summaries)

---

## 🗂️ Project Structure

```
📦 abstractive-summarization
├── 📓 NLP_Project.ipynb              # Full training notebook (all 5 models)
├── 🖥️  app.py                     # Streamlit demo app
├── 📄 README.md
│
├── 🤖 Models (download separately — see below)
│   ├── best_rnn_model.keras
│   ├── best_gru_model.keras
│   ├── best_lstm_model.keras
│   └── t5_finetuned/
│       ├── config.json
│       ├── tokenizer_config.json
│       └── pytorch_model.bin
│
└── 🔤 Tokenizers (download separately — see below)
    ├── article_tokenizer.pkl
    └── summary_tokenizer.pkl
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/abstractive-summarization.git
cd abstractive-summarization
```

### 2. Install dependencies
```bash
pip install tensorflow torch transformers datasets streamlit \
            nltk rouge-score numpy matplotlib scikit-learn
```

### 3. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 4. Download model files
Download the trained model files and place them in the project root:

- `best_rnn_model.keras`
- `best_gru_model.keras`
- `best_lstm_model.keras`
- `t5_finetuned/` (folder)
- `article_tokenizer.pkl`
- `summary_tokenizer.pkl`

⚠️ Model files are too large for GitHub. Download them via the Google Drive link > https://drive.google.com/drive/folders/1FmWBOmOo5jj5kxTNyFKkoyG_5iCAeO35?usp=sharing



## Running the Streamlit Demo

Make sure all model files are in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

Paste any news article and get summaries from all 5 models instantly, with the ROUGE score comparison table displayed below.

---


## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — RNN, GRU, LSTM
- **PyTorch + Hugging Face Transformers** — T5 fine-tuning
- **Hugging Face Datasets** — dataset loading
- **NLTK** — tokenization for HMM
- **rouge-score** — evaluation
- **Streamlit** — interactive demo
- **Google Colab** — training environment (free GPU)


---

## 📄 License

This project is for academic purposes as part of the Natural Language Processing Course.
