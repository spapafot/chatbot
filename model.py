from tensorflow.keras import layers, models


def create_embedding_layer(vocab_size, embedding_dim, max_len, embedding_matrix):
    embedding_layer = layers.Embedding(input_dim=vocab_size,
                                       output_dim=embedding_dim,
                                       input_length=max_len,
                                       weights=[embedding_matrix],
                                       trainable=False)
    return embedding_layer


def create_seq2seq_model(embedding_dim, vocab_size, max_len, embedding_layer):
    encoder_inputs = layers.Input(shape=(max_len,), dtype='int32')
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_LSTM = layers.LSTM(embedding_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = layers.Input(shape=(max_len,), dtype='int32')
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_LSTM = layers.LSTM(embedding_dim, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])

    outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(decoder_outputs)
    model = models.Model([encoder_inputs, decoder_inputs], outputs)

    return model
