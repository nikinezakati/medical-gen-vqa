from os import pardir
import numpy as np
from src.word2vec.model import Word2VecModel
from src.constants import WORD2VEC_MODEL_DIR, PREPROCCESSED_DATA_DIR, PREPROCESSED_DATA_SUFFIX, WORD2VEC_SIM_REPORTS_URL
from tasks import statistics
from word2vec.utils import save_csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--context", default=5)
args = parser.parse_args()


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) 

if __name__ == "__main__":
    total_names = ['Color','KG','Modality','Organ','Plane','Position','Shape','Size']
    C = args.context
    dummydim = 1
    common_tokens = statistics.common_tokens_relfreq(PREPROCCESSED_DATA_DIR)
    csv_columns = ["labels", "word", "cosine similarity"]
    csv_data = []
    for i in range(len(total_names)):
        for j in range(i + 1, len(total_names)):
            
            id1, id2 = total_names[i], total_names[j]
            class_1 = id1
            class_2 = id2
            data_url1 = os.path.join(PREPROCCESSED_DATA_DIR, f"{class_1}{PREPROCESSED_DATA_SUFFIX}.json")
            data_url2 = os.path.join(PREPROCCESSED_DATA_DIR, f"{class_2}{PREPROCESSED_DATA_SUFFIX}.json")
            model1 = Word2VecModel(C, dummydim, data_url1, class_1)
            model1.load_saved_params(WORD2VEC_MODEL_DIR)
            model2 = Word2VecModel(C, dummydim, data_url2, class_2)
            model2.load_saved_params(WORD2VEC_MODEL_DIR)
            words = common_tokens[(id1, id2)][:1]
            for w, refFreqScore in words:
                data = {}
                data[csv_columns[0]] = f"{class_1}-{class_2}"
                data[csv_columns[1]] = w
                v1 = model1.get_word_vector(w)
                v2 = model2.get_word_vector(w)
                similarity = cosine_similarity(v1, v2)
                data[csv_columns[2]] = round(similarity, 2)
                csv_data.append(data)
    save_csv(csv_columns, csv_data, WORD2VEC_SIM_REPORTS_URL)
