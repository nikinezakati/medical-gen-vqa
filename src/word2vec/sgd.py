#!/usr/bin/env python

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 5000

import pickle
import glob
import random
import numpy as np
import os
from src.constants import WORD2VEC_MODEL_DIR, WORD2VEC_SUFFIX
from src.logger.logger import log

def load_saved_params(class_name):

    params_file = os.path.join(WORD2VEC_MODEL_DIR, f"{class_name}.{WORD2VEC_SUFFIX}.npy")
    state_file = os.path.join(WORD2VEC_MODEL_DIR, f"state.{class_name}.{WORD2VEC_SUFFIX}.pickle")
    params = np.load(params_file)
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    return params, state


def save_params(class_name, params):
    params_file = os.path.join(WORD2VEC_MODEL_DIR, f"{class_name}.{WORD2VEC_SUFFIX}.npy")
    state_file = os.path.join(WORD2VEC_MODEL_DIR, f"state.{class_name}.{WORD2VEC_SUFFIX}.pickle")
    np.save(params_file, params)
    log(f"Saved model to {params_file}", "word2vec")
    with open(state_file, "wb") as f:
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, class_name, postprocessing=None, useSaved=False,
        PRINT_EVERY=10, LOG_EVERY=1000):

    print(f"[word2vec] Starting training on {class_name}.")
    log(f"[word2vec] Starting training on {class_name}.", 'word2vec')
    try:
        # Anneal learning rate every several iterations
        ANNEAL_EVERY = 20000

        start_iter = 0

        x = x0

        if not postprocessing:
            postprocessing = lambda x: x

        exploss = None

        for iter in range(start_iter + 1, iterations + 1):

            loss = None
            loss, grads = f(x)
            x = np.subtract(x, step * grads)

            x = postprocessing(x)
            if iter % PRINT_EVERY == 0:
                if not exploss:
                    exploss = loss
                else:
                    exploss = .95 * exploss + .05 * loss
                print("iter %d: %f" % (iter, exploss))
            if iter % LOG_EVERY == 0: 
                log("iter %d, Loss: %f" % (iter, exploss), "word2vec")

            if iter % ANNEAL_EVERY == 0:
                step *= 0.5
        print(f"Finished traingin on class {class_name} with {iterations} iters. Loss: {exploss}")
        log(f"Finished traingin on class {class_name} with {iterations} iters. Loss: {exploss}", "word2vec")
        return x
    except Exception as e:
        log(str(e), "word2vec")