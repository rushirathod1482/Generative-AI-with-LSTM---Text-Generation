import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. DATA PREPROCESSING

path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)

with open(path_to_file, "rb") as f:
    text = f.read().decode(
        encoding="utf-8").lower()[:60000]  # Subset for speed

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[: i + 1])

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. MODEL DESIGN

model = Sequential(
    [
        Embedding(total_words, 128, input_length=max_sequence_len - 1),
        LSTM(150, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dense(total_words, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.summary()

# 3. MODEL TRAINING & CHECKPOINTS

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(
        "best_lstm_model.h5", monitor="val_loss", save_best_only=True
    ),
]

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# 4. TEXT GENERATION LOGIC


def generate_text(seed_text, next_words=12, temperature=0.7):
    """Generates text based on seed input using temperature-based sampling."""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)[0]

        predictions = np.asarray(predictions).astype("float64")
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predictions), p=predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text


# 5. SAMPLE OUTPUT GENERATIONS

seed_inputs = ["to be or not", "wherefore art thou", "first citizen"]

for seed in seed_inputs:
    print(f"\nSeed Input: '{seed}'")
    print(f"Generated: {generate_text(seed, next_words=10, temperature=0.7)}")
