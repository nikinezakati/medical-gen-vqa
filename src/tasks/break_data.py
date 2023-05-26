import json
import sys
import re
from src.constants import PREPROCCESSED_DATA_DIR, BROKEN_WORD_DATA_DIR, BROKEN_SENT_DATA_DIR

def break_data(word_output_dir=BROKEN_WORD_DATA_DIR, sent_output_dir=BROKEN_SENT_DATA_DIR,):
    dir = PREPROCCESSED_DATA_DIR 

    train_dir = dir + '/train_en.json'
    validate_dir = dir + '/val_en.json'
    test_dir = dir + '/test_en.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    dct = { 'question_words': [], 'answer_words': []}
    for tr in train_data:
        result = re.findall(r"[\w']+", tr['question'])
        dct['question_words'].append(result)

        result = re.findall(r"[\w']+", tr['answer'])
        dct['answer_words'].append(result)

    j = json.dumps(dct)        
    with open(BROKEN_WORD_DATA_DIR+'/train.json', 'a') as fp:
        fp.write(j)
    
    train_data =[]


    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()
    dct = { 'question_words': [], 'answer_words': []}
    for tr in val_data:
        result = re.findall(r"[\w']+", tr['question'])
        dct['question_words'].append(result)

        result = re.findall(r"[\w']+", tr['answer'])
        dct['answer_words'].append(result)

    j = json.dumps(dct)        
    with open(BROKEN_WORD_DATA_DIR+'/val.json', 'a') as fp:
        fp.write(j)
    val_data =[]

    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    dct = { 'question_words': [], 'answer_words': []}
    for tr in test_data:
        result = re.findall(r"[\w']+", tr['question'])
        dct['question_words'].append(result)

        result = re.findall(r"[\w']+", tr['answer'])
        dct['answer_words'].append(result)

    j = json.dumps(dct)        
    with open(BROKEN_WORD_DATA_DIR+'/test.json', 'a') as fp:
        fp.write(j)
    test_data =[]

##############
    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    dct = { 'question_sents': []}
    for tr in train_data:
        lst=[]
        lst.append( tr['question'])
        dct['question_sents'].append(lst)

    j = json.dumps(dct)        
    with open(BROKEN_SENT_DATA_DIR+'/train.json', 'a') as fp:
        fp.write(j)
    train_data =[]


    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()
    dct = { 'question_sents': []}
    for tr in val_data:
        lst=[]
        lst.append( tr['question'])
        dct['question_sents'].append(lst)

    j = json.dumps(dct)        
    with open(BROKEN_SENT_DATA_DIR+'/val.json', 'a') as fp:
        fp.write(j)
    val_data =[]

    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    dct = { 'question_sents': []}
    for tr in test_data:
        lst=[]
        lst.append( tr['question'])
        dct['question_sents'].append(lst)

    j = json.dumps(dct)        
    with open(BROKEN_SENT_DATA_DIR+'/test.json', 'a') as fp:
        fp.write(j)
    test_data =[]    




if __name__ == "__main__":
    try:
        word_dir = sys.argv[1]
        sent_dir = sys.argv[2]
    except:
        word_dir = BROKEN_WORD_DATA_DIR
        sent_dir = BROKEN_SENT_DATA_DIR
    break_data(word_dir,sent_dir)   