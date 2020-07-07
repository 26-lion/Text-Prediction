import numpy as np
import random
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
text = "Final_dataset.txt"
te = open(text).read().lower()
maxlen = 30
steps = 3
sentences = []
next_chars = []
for i in range(0, len(te)-maxlen, steps):
    sentences.append(te[i:i+maxlen])
    next_chars.append(te[i+maxlen])
print("number of sequences", len(sentences))
chars = sorted(list(set(te)))
print("unique characters:", len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)
print("Vectorization")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


model = Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(layers.LSTM(128))
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="Adam")


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for epoch in range(1, 46):
    print("epoch", epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(te)-maxlen-1)
    generated_text = te[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.5, 0.8]:
        print("---temperature", temperature)
        sys.stdout.write(generated_text)
        for i in range(600):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)











