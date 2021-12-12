import ingest_data
import preprocessing
import model
import pandas as pd
import os

filepath = "/data"
all_files = os.walk(filepath)
character_names = ["αλεξης", "νταλια", "αφηγητρια", "ζουμπουλια", "αννα", "μαριλενα", "λία",
                   "ζουμπουλία", "άννα", "μαριλένα", "λια", "αλέξανδρος", "αβρααμ", "σταυριανιδης",
                   "ελενα", "σπυρος", "θεοπούλα", "θεοπουλα", "λεονταριδης"]

text_list = ingest_data.structure_files(all_files, filepath)
clean_list = ingest_data.create_clean_list(text_list, character_names)
VOCAB_SIZE = ingest_data.get_vocab_size(clean_list)
df = ingest_data.create_dataframe(clean_list)

MAX_LEN = 25
EMBEDDING_DIM = 300

X = df['input'].to_list()
y = df['output'].to_list()
data = X + y
word2idx, idx2word, vocab = preprocessing.vocab_creator(data, VOCAB_SIZE)
print(word2idx,vocab)


#embedding_layer = model.create_embedding_layer(vocab_size, embedding_dim, max_len, embedding_matrix)
#model = model.create_seq2seq_model(embedding_dim, vocab_size, max_len, embedding_layer)

#if __name__ == '__main__':
 #   pass
