import streamlit as st
import numpy as np
import pickle
import re
import tensorflow as tf
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

MAX_ARTICLE_LEN = 400
MAX_SUMMARY_LEN = 100
RNN_UNITS       = 256
PREFIX          = "summarize: "

def find_layer(model, name):
    """Finds layer by exact name first, then partial match — handles Keras auto-renaming"""
    try:
        return model.get_layer(name)
    except:
        for layer in model.layers:
            if name in layer.name:
                return layer
    raise ValueError(f"Layer '{name}' not found in model")

@st.cache_resource
def load_all_models():

    # ── Tokenizers ────────────────────────────────────
    with open('article_tokenizer.pkl', 'rb') as f:
        article_tokenizer = pickle.load(f)
    with open('summary_tokenizer.pkl', 'rb') as f:
        summary_tokenizer = pickle.load(f)

    start_token           = summary_tokenizer.word_index['sostok']
    end_token             = summary_tokenizer.word_index['eostok']
    reverse_summary_index = {v: k for k, v in summary_tokenizer.word_index.items()}

    # ── RNN ───────────────────────────────────────────
    print("Loading RNN...")
    rnn_model   = tf.keras.models.load_model('best_rnn_model.keras')

    inf_enc_rnn = Model(
        inputs  = rnn_model.inputs[0],
        outputs = find_layer(rnn_model, 'encoder_rnn').output[1]
    )

    dec_in_rnn    = Input(shape=(1,))
    dec_st_rnn    = Input(shape=(RNN_UNITS,))
    dec_emb_rnn   = find_layer(rnn_model, 'decoder_embedding')(dec_in_rnn)
    dec_out_rnn, dec_st_out_rnn = find_layer(rnn_model, 'decoder_rnn')(
        dec_emb_rnn, initial_state=dec_st_rnn)
    dec_dense_rnn = find_layer(rnn_model, 'output')(dec_out_rnn)
    inf_dec_rnn   = Model([dec_in_rnn, dec_st_rnn],
                           [dec_dense_rnn, dec_st_out_rnn])
    print("✅ RNN ready!")

    # ── GRU ───────────────────────────────────────────
    print("Loading GRU...")
    gru_model   = tf.keras.models.load_model('best_gru_model.keras')

    inf_enc_gru = Model(
        inputs  = gru_model.inputs[0],
        outputs = [find_layer(gru_model, 'encoder_gru').output[0],
                   find_layer(gru_model, 'encoder_gru').output[1]]
    )

    dec_in_gru     = Input(shape=(1,))
    enc_out_in_gru = Input(shape=(MAX_ARTICLE_LEN, RNN_UNITS))
    dec_st_in_gru  = Input(shape=(RNN_UNITS,))
    dec_emb_gru    = find_layer(gru_model, 'decoder_embedding')(dec_in_gru)
    dec_out_gru, dec_st_out_gru = find_layer(gru_model, 'decoder_gru')(
        dec_emb_gru, initial_state=dec_st_in_gru)
    att_out_gru    = find_layer(gru_model, 'attention_layer')(
        [dec_out_gru, enc_out_in_gru])
    cat_out_gru    = find_layer(gru_model, 'concat')(
        [dec_out_gru, att_out_gru])
    dense_out_gru  = find_layer(gru_model, 'output')(cat_out_gru)
    inf_dec_gru    = Model([dec_in_gru, enc_out_in_gru, dec_st_in_gru],
                            [dense_out_gru, dec_st_out_gru])
    print("✅ GRU ready!")

    # ── LSTM ──────────────────────────────────────────
    print("Loading LSTM...")
    lstm_model   = tf.keras.models.load_model('best_lstm_model.keras')

    inf_enc_lstm = Model(
        inputs  = lstm_model.inputs[0],
        outputs = [find_layer(lstm_model, 'encoder_lstm').output[0],
                   find_layer(lstm_model, 'encoder_lstm').output[1],
                   find_layer(lstm_model, 'encoder_lstm').output[2]]
    )

    dec_in_lstm     = Input(shape=(1,))
    enc_out_in_lstm = Input(shape=(MAX_ARTICLE_LEN, RNN_UNITS))
    dec_h_in_lstm   = Input(shape=(RNN_UNITS,))
    dec_c_in_lstm   = Input(shape=(RNN_UNITS,))
    dec_emb_lstm    = find_layer(lstm_model, 'decoder_embedding')(dec_in_lstm)
    dec_out_lstm, dec_h_out, dec_c_out = find_layer(lstm_model, 'decoder_lstm')(
        dec_emb_lstm, initial_state=[dec_h_in_lstm, dec_c_in_lstm])
    att_out_lstm    = find_layer(lstm_model, 'attention_layer')(
        [dec_out_lstm, enc_out_in_lstm])
    cat_out_lstm    = find_layer(lstm_model, 'concat')(
        [dec_out_lstm, att_out_lstm])
    dense_out_lstm  = find_layer(lstm_model, 'output')(cat_out_lstm)
    inf_dec_lstm    = Model(
        [dec_in_lstm, enc_out_in_lstm, dec_h_in_lstm, dec_c_in_lstm],
        [dense_out_lstm, dec_h_out, dec_c_out]
    )
    print("✅ LSTM ready!")

    # ── T5 ────────────────────────────────────────────
    print("Loading T5...")
    t5_model     = T5ForConditionalGeneration.from_pretrained('./t5_finetuned')
    t5_tokenizer = AutoTokenizer.from_pretrained('./t5_finetuned')
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model     = t5_model.to(device)
    t5_model.eval()
    print("✅ T5 ready!")

    return (article_tokenizer, summary_tokenizer,
            start_token, end_token, reverse_summary_index,
            inf_enc_rnn,  inf_dec_rnn,
            inf_enc_gru,  inf_dec_gru,
            inf_enc_lstm, inf_dec_lstm,
            t5_model, t5_tokenizer, device)

# ── Helper Functions ─────────────────────────────────
def preprocess_article(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def encode_article(text, article_tokenizer):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = article_tokenizer.texts_to_sequences([preprocess_article(text)])
    return pad_sequences(seq, maxlen=MAX_ARTICLE_LEN,
                         padding='post', truncating='post')

def decode_tokens(ids, reverse_summary_index, start_token, end_token):
    words = [reverse_summary_index.get(i, '')
             for i in ids if i not in [0, start_token, end_token]]
    return ' '.join(words).strip() or "Could not generate summary."

# ── Summary Functions ────────────────────────────────
def summarize_rnn(article, models):
    article_tokenizer, _, start_token, end_token, reverse_summary_index, \
    inf_enc_rnn, inf_dec_rnn, *_ = models
    seq    = encode_article(article, article_tokenizer)
    state  = inf_enc_rnn.predict(seq, verbose=0)
    target, result = np.array([[start_token]]), []
    for _ in range(MAX_SUMMARY_LEN - 1):
        out, state = inf_dec_rnn.predict([target, state], verbose=0)
        wid = np.argmax(out[0, 0, :])
        if wid == end_token: break
        result.append(wid)
        target = np.array([[wid]])
    return decode_tokens(result, reverse_summary_index, start_token, end_token)

def summarize_gru(article, models):
    article_tokenizer, _, start_token, end_token, reverse_summary_index, \
    _, _, inf_enc_gru, inf_dec_gru, *_ = models
    seq = encode_article(article, article_tokenizer)
    enc_out, state = inf_enc_gru.predict(seq, verbose=0)
    target, result = np.array([[start_token]]), []
    for _ in range(MAX_SUMMARY_LEN - 1):
        out, state = inf_dec_gru.predict([target, enc_out, state], verbose=0)
        wid = np.argmax(out[0, 0, :])
        if wid == end_token: break
        result.append(wid)
        target = np.array([[wid]])
    return decode_tokens(result, reverse_summary_index, start_token, end_token)

def summarize_lstm(article, models):
    article_tokenizer, _, start_token, end_token, reverse_summary_index, \
    _, _, _, _, inf_enc_lstm, inf_dec_lstm, *_ = models
    seq = encode_article(article, article_tokenizer)
    enc_out, h, c = inf_enc_lstm.predict(seq, verbose=0)
    target, result = np.array([[start_token]]), []
    for _ in range(MAX_SUMMARY_LEN - 1):
        out, h, c = inf_dec_lstm.predict([target, enc_out, h, c], verbose=0)
        wid = np.argmax(out[0, 0, :])
        if wid == end_token: break
        result.append(wid)
        target = np.array([[wid]])
    return decode_tokens(result, reverse_summary_index, start_token, end_token)

def summarize_hmm(article):
    stop_words = set(stopwords.words('english'))
    sentences  = sent_tokenize(article)
    if len(sentences) <= 3:
        return ' '.join(sentences)
    freq = {}
    for w in word_tokenize(article.lower()):
        if w.isalpha() and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1
    scored = [(sum(freq.get(w, 0) for w in word_tokenize(s.lower())
                   if w.isalpha()) / (len(word_tokenize(s)) + 1e-9), i, s)
              for i, s in enumerate(sentences)]
    top = sorted(sorted(scored, reverse=True)[:3], key=lambda x: x[1])
    return ' '.join([s for _, _, s in top])

def summarize_t5(article, models):
    *_, t5_model, t5_tokenizer, device = models
    inputs = t5_tokenizer(
        PREFIX + article,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    ).to(device)
    with torch.no_grad():
        output = t5_model.generate(
            inputs['input_ids'],
            attention_mask       = inputs['attention_mask'],
            max_length           = 150,
            min_length           = 30,
            num_beams            = 4,
            early_stopping       = True,
            no_repeat_ngram_size = 3
        )
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# ── Streamlit UI ─────────────────────────────────────
st.set_page_config(page_title="News Summarizer", layout="wide", page_icon="📰")

st.title("News Article Summarizer")
st.markdown("### Compare RNN · GRU · LSTM · HMM · T5 Transformer")
st.markdown("Paste a news article below and generate summaries from all 5 models.")

with st.spinner("Loading all models... (first run only, ~1 min)"):
    models = load_all_models()
st.success("All models loaded and ready!")

article = st.text_area("Paste Your News Article Here",
                        height=200,
                        placeholder="Paste a news article here...")

if st.button("Generate Summaries", type="primary"):
    if not article.strip():
        st.warning("Please paste an article first!")
    else:
        st.markdown("## Generated Summaries")
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Generating RNN summary..."):
                st.markdown("#### 🔴 Vanilla RNN")
                st.info(summarize_rnn(article, models))

            with st.spinner("Generating GRU summary..."):
                st.markdown("#### 🟠 GRU + Attention")
                st.info(summarize_gru(article, models))

            with st.spinner("Generating LSTM summary..."):
                st.markdown("#### 🟡 LSTM + Attention")
                st.info(summarize_lstm(article, models))

        with col2:
            with st.spinner("Generating HMM summary..."):
                st.markdown("#### 🟢 HMM (Extractive)")
                st.info(summarize_hmm(article))

            with st.spinner("Generating T5 summary..."):
                st.markdown("#### 🔵 T5 Transformer (Best)")
                st.success(summarize_t5(article, models))

        st.markdown("---")
        st.markdown("### 📊 Model Performance (ROUGE Scores)")
        st.table({
            "Model"  : ["Vanilla RNN", "GRU", "LSTM", "HMM", "T5 Transformer"],
            "ROUGE-1": [0.0110, 0.1214, 0.1420, 0.2076, 0.3590],
            "ROUGE-2": [0.0001, 0.0182, 0.0259, 0.0569, 0.1229],
            "ROUGE-L": [0.0091, 0.0916, 0.1119, 0.1189, 0.1953],
        })