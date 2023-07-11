from src.constants import WORD2VEC_LOG_FILE
import os
import datetime
def log(message, task):
    if(task == "word2vec"):
        if (not os.path.exists(WORD2VEC_LOG_FILE)):
            with open(WORD2VEC_LOG_FILE, 'w') as f:
                f.write(f"[{datetime.datetime.now()}] {message}\n")
        else:
            with open(WORD2VEC_LOG_FILE, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] {message}\n")