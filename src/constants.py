import os

BASE_DATA_URL =  "data" 
STATS_DATA_DIR = "stats"
RAW_DATA_DIR = os.path.join(BASE_DATA_URL, 'raw') 
PREPROCCESSED_DATA_DIR = os.path.join(BASE_DATA_URL, 'preprocessed') 
BROKEN_WORD_DATA_DIR = os.path.join(BASE_DATA_URL, 'wordbroken') 
BROKEN_SENT_DATA_DIR = os.path.join(BASE_DATA_URL, 'sentencebroken') 
FULL_SENTENCE_DATA_DIR = os.path.join(BASE_DATA_URL, 'full_sentence_data') 
