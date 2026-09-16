import streamlit as st
import html
import pickle
import numpy as np
from st_keyup import st_keyup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("LSTM_model2.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()
print("max_len:", max_len)
# ------------------------------
# Prediction function
# ------------------------------
def predict_next_words(text, word_count):
    sequence = tokenizer.texts_to_sequences([text])[0]
    predicted_words = []

    for _ in range(word_count):
        padded_sequence = pad_sequences(
            [sequence], maxlen=max_len - 1, padding="pre"
        )
        preds = model.predict(padded_sequence, verbose=0)
        predicted_index = int(np.argmax(preds))
        predicted_word = tokenizer.index_word.get(predicted_index)

        if not predicted_word:
            break
        predicted_words.append(predicted_word)
        sequence.append(predicted_index)

    return predicted_words

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Start typing and pause to receive a prediction.")

word_count = st.slider(
    
    "Number of words to predict",
    min_value=1,
    max_value=20,
    value=5,
    
)
user_input = st_keyup(
    "✍️ Enter text:",
    placeholder="Type a sentence here...",
    debounce=700,
    key="user_input",
)


suggestion_placeholder = st.empty()

if user_input and user_input.strip():
    safe_user_input = html.escape(user_input)
    suggestion_placeholder.markdown(
        f"""
        <div style="
            margin-top: 10px;
            padding: 14px;
            border-left: 4px solid #888;
            background-color: rgba(128, 128, 128, 0.12);
            border-radius: 6px;
            font-size: 20px;
            color: #888;
        ">
            <strong>{safe_user_input} ...</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    predicted_words = predict_next_words(user_input, word_count)
    suggestion = html.escape(" ".join(predicted_words)) if predicted_words else "..."
    suggestion_placeholder.markdown(
        f"""
        <div style="
            margin-top: 10px;
            padding: 14px;
            border-left: 4px solid #888;
            background-color: rgba(128, 128, 128, 0.12);
            border-radius: 6px;
            font-size: 20px;
        ">
           {safe_user_input} <span style="color: #888;">{suggestion}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    suggestion_placeholder.markdown(
        """
        <div style="
                   margin-top: 10px;
                   padding: 14px;
                   border-left: 4px solid #888;
                   background-color: rgba(128, 128, 128, 0.12);
                   border-radius: 6px;
                   font-size: 20px;
                   fopnt-style: italic;
                   color: #888;
               ">
                  Your prediction will appear here as you type...
               </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Model Performance")
accuracy_column, loss_column = st.columns(2)

with accuracy_column:
    st.metric("Model Accuracy", "88.45%")

with loss_column:
    st.metric("Model Loss", "0.5175")

st.image("Model Accuracy LSTM.jpg", caption="LSTM model accuracy", use_container_width=True)




# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")