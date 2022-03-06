# Greek Chatbot

Attempt to train a vanilla LSTM chatbot on a dataset of 50k greek subtitle pairs

After standard preprocessing 

1. Clean,
2. Add <START> and <END> tags
3. Create vocabulary
4. Tokenize
5. Pad sequences 
6. Create embedding matrix
  
### IT FAILED

> MemoryError: Unable to allocate 23.5 GiB for an array with shape (12092, 25, 20828) and data type float32

After running on a TPU in Google Colab i managed to train for some epochs but still crashed on save checkpoint
  
I will keep it here as a case study, the working version is at https://github.com/spapafot/greek_chatbot_with_attention
  
I added attention mechanism and used **tensorflow.data.Dataset** for optimization 
