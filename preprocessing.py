import string
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import el_core_news_md

nlp = el_core_news_md.load()
MAX_LEN = 25
VOCAB_SIZE = 20000
EMBEDDING_DIM = 300


def clean_text(text):

    d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = unicodedata.normalize('NFD', text).translate(d)

    return text

cleaned_text = clean_text('ήξερα,  7 το Κάνουμε ήξερα  4 τι να Κάνουμε τώρα?')
print(cleaned_text)


def tag_start_end_sentences(decoder_input_sentence):
    bos = "<BOS>"
    eos = "<EOS>"
    final_target = [bos + text + eos for text in decoder_input_sentence]
    return final_target

text = tag_start_end_sentences([cleaned_text])
print(text)


def vocab_creator(text_lists, VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(text_lists)
    dictionary = tokenizer.word_index

    word2idx = {}
    idx2word = {}
    vocab = []

    for k, v in dictionary.items():
        if v < VOCAB_SIZE:
            word2idx[k] = v
            idx2word[v] = k
            vocab.append(k)

        if v >= VOCAB_SIZE - 1:
            continue

    return word2idx, idx2word, vocab


word2idx, idx2word, vocab = vocab_creator(text, 10)
print(word2idx, "\n", idx2word)
print(f"vocab: {vocab}")


def text2seq(encoder_text, decoder_text, VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(encoder_text)
    tokenizer.fit_on_texts(decoder_text)
    encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

    return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq([cleaned_text], [cleaned_text], 50)
print(f"encoder_sequences: {encoder_sequences} \n decoder_sequences: {decoder_sequences}")


def padding(encoder_sequences, decoder_sequences, MAX_LEN):
    encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

    return encoder_input_data, decoder_input_data


encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, 20)
print(f"encoder_input_data:{encoder_input_data} \ndecoder_input_data:{decoder_input_data}")


def create_embedding_matrix(vocab):
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for i, word in enumerate(vocab):
        embedding_matrix[i] = nlp(word).vector
    return embedding_matrix


embedding_matrix = create_embedding_matrix(vocab)
print(embedding_matrix[0])


def create_decoder_output(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
    decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if j > 0:
                decoder_output_data[i][j][seq] = 1.

    return decoder_output_data


decoder_output_data = create_decoder_output(decoder_input_data, len(encoder_sequences), MAX_LEN, 10)

print(decoder_output_data.shape)
print(len(vocab))
