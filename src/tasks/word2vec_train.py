import argparse
import json
import os
import random
import numpy as np
from src.word2vec.model import Word2VecModel
from src.constants import PREPROCCESSED_DATA_DIR, CLASSIFIED_DATA_DIR


parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="1,2,3,4,5,6,7,8,-1")
parser.add_argument("--dim", default=10)
parser.add_argument("--context", default=5)
parser.add_argument("--iters", default=20000)
parser.add_argument("--seed", default=31415)

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)

if __name__ == "__main__":
    classes = args.names
    C = args.context
    dimVectors = args.dim
    iters = args.iters
    classes = [int(item) for item in classes.split(',')]
    for name in classes:
        if(id != -1):
            class_name = name
            data_url = os.path.join(CLASSIFIED_DATA_DIR, f"{class_name}.json")
        else:
            class_name = "all"
            data_url = os.path.join(PREPROCCESSED_DATA_DIR, "train_en.json")
        model = Word2VecModel(C, dimVectors, data_url, class_name)
        model.train(iters)
        model.save()
