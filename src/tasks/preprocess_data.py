import json
import sys
from src.constants import PREPROCCESSED_DATA_DIR, RAW_DATA_DIR

def preprocess_data(eng_data_output_dir=PREPROCCESSED_DATA_DIR):
    dir = RAW_DATA_DIR + '/Slake1.0/'

    train_dir = dir + 'train.json'
    validate_dir = dir + 'validate.json'
    test_dir = dir + 'test.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    train_eng=[]
    for tr in train_data:
        if tr['q_lang'] == 'en':
           train_eng.append(tr)       
    j = json.dumps(train_eng)        
    with open(PREPROCCESSED_DATA_DIR+'/train_en.json', 'a') as fp:
        fp.write(j)
    train_data =[]

    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()    
    val_eng=[]
    for v in val_data:
        if v['q_lang'] == 'en':
            val_eng.append(v)
    j = json.dumps(val_eng)     
    with open(PREPROCCESSED_DATA_DIR+'/val_en.json', 'a') as fp:
        fp.write(j)
    val_data =[]   

    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()    
    test_eng=[]
    for te in test_data:
        if te['q_lang'] == 'en':
            test_eng.append(te)
    j = json.dumps(test_eng)     
    with open(PREPROCCESSED_DATA_DIR+'/test_en.json', 'a') as fp:
        fp.write(j)
    test_data =[]          





if __name__ == "__main__":
    try:
        dir = sys.argv[1]
    except:
        dir = RAW_DATA_DIR
    preprocess_data(dir)        

