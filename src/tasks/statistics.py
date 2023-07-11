import json
import sys
import re
from src.constants import RAW_DATA_DIR, PREPROCCESSED_DATA_DIR, BROKEN_WORD_DATA_DIR, STATS_DATA_DIR
import csv
import matplotlib.pyplot as plt
import pandas as pd

def data_count(broken_word_dir=BROKEN_WORD_DATA_DIR):

    #raw
    dir = RAW_DATA_DIR + '/Slake1.0/'

    train_dir = dir + 'train.json'
    validate_dir = dir + 'validate.json'
    test_dir = dir + 'test.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    raw_train_count = len(train_data)
    dct = {'raw_train_count':raw_train_count}
    
    with open(STATS_DATA_DIR+'/raw_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Data","Count"])    
        for key, value in dct.items():
            writer.writerow([key, value])

    f = open(validate_dir, encoding="utf8")
    valid_data = json.load(f)
    f.close()
    raw_valid_count = len(valid_data)
    dct = {'raw_valid_count':raw_valid_count}

    with open(STATS_DATA_DIR+'/raw_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])

    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    raw_test_count = len(test_data)
    dct = {'raw_test_count': raw_test_count}

    with open(STATS_DATA_DIR+'/raw_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


    #english

    dir = PREPROCCESSED_DATA_DIR 

    train_dir = dir + '/train_en.json'
    validate_dir = dir + '/val_en.json'
    test_dir = dir + '/test_en.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    english_train_count = len(train_data)
    dct = {'english_train_count':english_train_count}

    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Sentence","Count"])    
        for key, value in dct.items():
            writer.writerow([key, value])

    f = open(validate_dir, encoding="utf8")
    valid_data = json.load(f)
    f.close()
    english_valid_count = len(valid_data)
    dct = {'english_valid_count':english_valid_count}

    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])


    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    english_test_count = len(test_data)
    dct = {'english_test_count': english_test_count}

    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


    #unique questions
    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()

    unique_dict=[]
    unique_questions = []
    unique_questions_cnt= {}
    for d in train_data:
        if d['question'] not in unique_questions:
            unique_questions.append(d['question'])
            unique_dict.append(d)

        if d['question'] not in unique_questions_cnt:
            unique_questions_cnt[d['question']]=1
        else:
            unique_questions_cnt[d['question']]+=1

    unique_questions_cnt = dict(sorted(unique_questions_cnt.items(), key=lambda item: item[1],reverse=True))        

    dct = {'train_uniqe_question_count': len(unique_dict)}
    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp)   
        for key, value in dct.items():
            writer.writerow([key, value])

    with open(STATS_DATA_DIR+'/train_question_unique_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Question","Count"])    
        for key, value in unique_questions_cnt.items():
            writer.writerow([key, value])        


    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()

    unique_dict=[]
    unique_questions = []
    unique_questions_cnt= {}
    for d in val_data:
        if d['question'] not in unique_questions:
            unique_questions.append(d['question'])
            unique_dict.append(d)

        if d['question'] not in unique_questions_cnt:
            unique_questions_cnt[d['question']]=1
        else:
            unique_questions_cnt[d['question']]+=1

    unique_questions_cnt = dict(sorted(unique_questions_cnt.items(), key=lambda item: item[1],reverse=True))             

    dct = {'val_uniqe_question_count': len(unique_dict)}
    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        for key, value in dct.items():
            writer.writerow([key, value])

    with open(STATS_DATA_DIR+'/val_question_unique_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Question","Count"])    
        for key, value in unique_questions_cnt.items():
            writer.writerow([key, value])     




    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()

    unique_dict=[]
    unique_questions = []
    unique_questions_cnt= {}
    for d in test_data:
        if d['question'] not in unique_questions:
            unique_questions.append(d['question'])
            unique_dict.append(d)

        if d['question'] not in unique_questions_cnt:
            unique_questions_cnt[d['question']]=1
        else:
            unique_questions_cnt[d['question']]+=1

    unique_questions_cnt = dict(sorted(unique_questions_cnt.items(), key=lambda item: item[1],reverse=True))             

    dct = {'test_uniqe_question_count': len(unique_dict)}
    with open(STATS_DATA_DIR+'/sentence_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])

    with open(STATS_DATA_DIR+'/test_question_unique_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Question","Count"])    
        for key, value in unique_questions_cnt.items():
            writer.writerow([key, value])                         
        


             

    #word

    dir = BROKEN_WORD_DATA_DIR 

    train_dir = dir + '/train.json'
    validate_dir = dir + '/val.json'
    test_dir = dir + '/test.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    all_question_words=train_data['question_words']
    train_qword_count=0
    uniqe_train_question=[]
    for lst in all_question_words:
        for word in lst:
            train_qword_count+=1
            if word not in uniqe_train_question:
                uniqe_train_question.append(word)
       
    dct = {'train_qword_count': train_qword_count}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in dct.items():
            writer.writerow([key, value])

    uniqe_train_question_cnt= len(uniqe_train_question)
    dct = {'uniqe_train_question_cnt': uniqe_train_question_cnt}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])



    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()
    all_question_words=val_data['question_words']
    val_qword_count=0
    uniqe_val_question=[]
    for lst in all_question_words:
        for word in lst:
            val_qword_count+=1
            if word not in uniqe_val_question:
                uniqe_val_question.append(word)
       
    dct = {'val_qword_count': val_qword_count}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])

    uniqe_val_question_cnt= len(uniqe_val_question)
    dct = {'uniqe_val_question_cnt': uniqe_val_question_cnt}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])



    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    all_question_words=test_data['question_words']
    test_qword_count=0
    uniqe_test_question=[]
    for lst in all_question_words:
        for word in lst:
            test_qword_count+=1
            if word not in uniqe_test_question:
                uniqe_test_question.append(word)
       
    dct = {'test_qword_count': test_qword_count} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


    uniqe_test_question_cnt= len(uniqe_test_question)
    dct = {'uniqe_test_question_cnt': uniqe_test_question_cnt}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])


    unique_train = {}
    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    all_question_words=train_data['answer_words']
    train_aword_count=0
    uniqe_train_answer = []
    uniqe_train_answer_whole = []
    for lst in all_question_words:
        for word in lst:
            train_aword_count+=1
            if word not in uniqe_train_answer:
                uniqe_train_answer.append(word) 
        
            if word in uniqe_train_answer:
                tmp=[]
                tmp.append(word)
                uniqe_train_answer_whole.append(tmp)
                if word not in unique_train:
                    unique_train[word] = 1
                else:
                    unique_train[word] += 1 

    unique_train = dict(sorted(unique_train.items(), key=lambda item: item[1],reverse=True))        
                
    with open(STATS_DATA_DIR+'/unique_train.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])       
        for key, value in unique_train.items():
            writer.writerow([key, value])      
                   
       
    dct = {'train_aword_count':train_aword_count} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])

    uniqe_train_answer_cnt= len(uniqe_train_answer)
    dct = {'uniqe_train_answer_cnt': uniqe_train_answer_cnt} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)   
        for key, value in dct.items():
            writer.writerow([key, value])


    unique_val = {}
    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()
    all_question_words=val_data['answer_words']
    val_aword_count=0
    uniqe_val_answer=[]
    uniqe_val_answer_whole=[]
    for lst in all_question_words:
        for word in lst:
            val_aword_count+=1
            if word not in uniqe_val_answer:
                uniqe_val_answer.append(word)
                

            if word in uniqe_val_answer:
                tmp=[]
                tmp.append(word)
                uniqe_val_answer_whole.append(tmp)    

                if word not in unique_val:
                    unique_val[word] = 1
                else:
                    unique_val[word] += 1  

    unique_val = dict(sorted(unique_val.items(), key=lambda item: item[1],reverse=True))    
    with open(STATS_DATA_DIR+'/unique_val.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in unique_val.items():
            writer.writerow([key, value])                 
       
    dct = {'val_aword_count': val_aword_count} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])

    uniqe_val_answer_cnt= len(uniqe_val_answer)
    dct = {'uniqe_val_answer_cnt': uniqe_val_answer_cnt} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])



    unique_test = {}
    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    all_question_words=test_data['answer_words']
    test_aword_count=0
    uniqe_test_answer=[]
    uniqe_test_answer_whole=[]
    for lst in all_question_words:
        for word in lst:
            test_aword_count+=1
            if word not in uniqe_test_answer:
                uniqe_test_answer.append(word)


            if word in uniqe_test_answer:
                tmp=[]
                tmp.append(word)
                uniqe_test_answer_whole.append(tmp)   

                if word not in unique_test:
                    unique_test[word] = 1
                else:
                    unique_test[word] += 1  

    unique_test = dict(sorted(unique_test.items(), key=lambda item: item[1],reverse=True))  
    with open(STATS_DATA_DIR+'/unique_test.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in unique_test.items():
            writer.writerow([key, value])                    
       
    dct = {'test_aword_count': test_aword_count} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


    uniqe_test_answer_cnt= len(uniqe_test_answer)
    dct = {'uniqe_test_answer_cnt': uniqe_test_answer_cnt} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])    

        
    return uniqe_train_answer_whole, uniqe_val_answer_whole, uniqe_test_answer_whole, unique_train, unique_val, unique_test



def common_tokens(uniqe_train_answer, uniqe_val_answer, uniqe_test_answer):    
    #train

    common_unique_train = {}
    uncommon_unique_train = {} 
    common_unique_train_count = 0
    uncommon_unique_train_count = 0
    for i in range(len(uniqe_train_answer)):
        for j in range(i + 1 , len(uniqe_train_answer)):
            for word in uniqe_train_answer[i]:
                if word not in common_unique_train:
                    common_unique_train[word] = uniqe_train_answer[j].count(word)
                else:
                    common_unique_train[word] += uniqe_train_answer[j].count(word)              

    for key, value in common_unique_train.copy().items():
        if value == 0:
            uncommon_unique_train_count+=1
            uncommon_unique_train[key] = value 
            common_unique_train.pop(key)
        else:
            common_unique_train_count+=1  

    common_unique_train = dict(sorted(common_unique_train.items(), key=lambda item: item[1],reverse=True))          

    with open(STATS_DATA_DIR+'/common_unique_train.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in common_unique_train.items():
            writer.writerow([key, value])             

    with open(STATS_DATA_DIR+'/uncommon_unique_train.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in uncommon_unique_train.items():
            writer.writerow([key, value])    

    
    dct = {'common_unique_train_count': common_unique_train_count} 

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)   
        for key, value in dct.items():
            writer.writerow([key, value])


    dct = {'uncommon_unique_train_count': uncommon_unique_train_count}

    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


    #val
    common_unique_val = {}
    uncommon_unique_val = {}
    common_unique_val_count = 0
    uncommon_unique_val_count = 0
    for i in range(len(uniqe_val_answer)):
        for j in range(i + 1 , len(uniqe_val_answer)):
            for word in uniqe_val_answer[i]:
                if word not in common_unique_val:
                    common_unique_val[word] = uniqe_val_answer[j].count(word)
                else:
                    common_unique_val[word] += uniqe_val_answer[j].count(word)              
    
    for key, value in common_unique_val.copy().items():
        if value == 0:
            uncommon_unique_val_count+=1
            uncommon_unique_val[key] = value
            common_unique_val.pop(key)

        else:
            common_unique_val_count+=1  

    common_unique_val = dict(sorted(common_unique_val.items(), key=lambda item: item[1],reverse=True)) 

    with open(STATS_DATA_DIR+'/common_unique_val.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in common_unique_val.items():
            writer.writerow([key, value])        

    with open(STATS_DATA_DIR+'/uncommon_unique_val.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in uncommon_unique_val.items():
            writer.writerow([key, value])        
    
    dct = {'common_unique_val_count':common_unique_val_count} 
    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp) 
        for key, value in dct.items():
            writer.writerow([key, value])

    dct = {'uncommon_unique_val_count': uncommon_unique_val_count} 
    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)   
        for key, value in dct.items():
            writer.writerow([key, value])

    #test
    common_unique_test = {}
    uncommon_unique_test = {}
    common_unique_test_count = 0
    uncommon_unique_test_count = 0
    for i in range(len(uniqe_test_answer)):
        for j in range(i + 1 , len(uniqe_test_answer)):
            for word in uniqe_test_answer[i]:
                if word not in common_unique_test:
                    common_unique_test[word] = uniqe_test_answer[j].count(word)
                else:
                    common_unique_test[word] += uniqe_test_answer[j].count(word)

    for key, value in common_unique_test.copy().items():
        if value == 0:
            uncommon_unique_test_count+=1
            uncommon_unique_test[key] = value 
            common_unique_test.pop(key)
            
        else:
            common_unique_test_count+=1 

    common_unique_test = dict(sorted(common_unique_test.items(), key=lambda item: item[1],reverse=True)) 

    with open(STATS_DATA_DIR+'/common_unique_test.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in common_unique_test.items():
            writer.writerow([key, value])             

    with open(STATS_DATA_DIR+'/uncommon_unique_test.csv', 'a') as fp:
        writer = csv.writer(fp)
        writer.writerow(["Word","Count"])    
        for key, value in uncommon_unique_test.items():
            writer.writerow([key, value])         
    
    dct = {'common_unique_test_count':common_unique_test_count} 
    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)
        for key, value in dct.items():
            writer.writerow([key, value])

    dct = {'uncommon_unique_test_count':uncommon_unique_test_count} 
    with open(STATS_DATA_DIR+'/word_count.csv', 'a') as fp:
        writer = csv.writer(fp)  
        for key, value in dct.items():
            writer.writerow([key, value])


def draw_histogram(uniqe_train_answer, uniqe_val_answer, uniqe_test_answer):   
    
    #train 

    x = []
    y = []

    with open(STATS_DATA_DIR+'/unique_train.csv','r') as csvfile:
        df = pd.read_csv(csvfile) 

    text = pd.DataFrame(df, columns=["Word","Count"])    
      
    i = 0
    for w in text['Word']:
        if i<21:
            x.append(w)
        else:
           break
        i+=1  

    i = 0    
    for c in text['Count']:
        if i<21:
            y.append(c)  
        else:
            break       
        i+=1     
  
    plt.figure(figsize=(20,10))
    plt.bar(x, y, color = 'g', width = 0.72, label = "Occurrence")
    plt.ylabel('Occurrence')
    plt.title('Occurrence of Unique Train Words')
    plt.savefig(STATS_DATA_DIR+'/train_unique_words.png')


    #val
    x = []
    y = []

    with open(STATS_DATA_DIR+'/unique_val.csv','r') as csvfile:
        df = pd.read_csv(csvfile) 

    text = pd.DataFrame(df, columns=["Word","Count"])    
      
    i = 0
    for w in text['Word']:
        if i<21:
            x.append(w)
        else:
           break
        i+=1  

    i = 0    
    for c in text['Count']:
        if i<21:
            y.append(c)  
        else:
            break       
        i+=1     
  
    plt.figure(figsize=(20,10))
    plt.bar(x, y, color = 'g', width = 0.72, label = "Occurrence")
    plt.ylabel('Occurrence')
    plt.title('Occurrence of Unique Val Words')
    plt.savefig(STATS_DATA_DIR+'/val_unique_words.png')



    #test
    x = []
    y = []

    with open(STATS_DATA_DIR+'/unique_test.csv','r') as csvfile:
        df = pd.read_csv(csvfile) 

    text = pd.DataFrame(df, columns=["Word","Count"])    
      
    i = 0
    for w in text['Word']:
        if i<21:
            x.append(w)
        else:
           break
        i+=1  

    i = 0    
    for c in text['Count']:
        if i<21:
            y.append(c)  
        else:
            break       
        i+=1     
  
    plt.figure(figsize=(20,10))
    plt.bar(x, y, color = 'g', width = 0.72, label = "Occurrence")
    plt.ylabel('Occurrence')
    plt.title('Occurrence of Unique Test Words')
    plt.savefig(STATS_DATA_DIR+'/test_unique_words.png')

   





if __name__ == "__main__":
    uniqe_train_answer_whole, uniqe_val_answer_whole,uniqe_test_answer_whole, unique_train, unique_val, unique_test  = data_count() 
    common_tokens(uniqe_train_answer_whole, uniqe_val_answer_whole,uniqe_test_answer_whole) 
    draw_histogram(unique_train, unique_val, unique_test)


