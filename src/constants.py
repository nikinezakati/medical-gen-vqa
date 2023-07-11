import os
BASE_DIR = "./"
DATA_DIR = "dataset"

STATS_DATA_DIR = "stats"
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') 
PREPROCCESSED_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed') 
CLASSIFIED_DATA_DIR = os.path.join(DATA_DIR, 'classified/') 
BROKEN_WORD_DATA_DIR = os.path.join(DATA_DIR, 'wordbroken') 
BROKEN_SENT_DATA_DIR = os.path.join(DATA_DIR, 'sentencebroken') 
FULL_SENTENCE_DATA_DIR = os.path.join(DATA_DIR, 'full_sentence_data/') 
IMGS_DIR = os.path.join(RAW_DATA_DIR, 'Slake1.0/imgs/') 
IMGS_RESIZED_DIR =  os.path.join(DATA_DIR, 'imgs_resized/') 
IMG_FEATTURES_DIR =  os.path.join(DATA_DIR, 'imgs/') 
LOGS_DIR = os.path.join(DATA_DIR, 'logs/') 
PKLS_DIR = os.path.join(DATA_DIR, 'pickles/') 