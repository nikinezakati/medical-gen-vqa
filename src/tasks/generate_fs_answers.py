import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('popular')
import json
import re
from pattern.en import conjugate, PAST, PRESENT, tenses, parse, pprint, parsetree, SINGULAR, PLURAL, pluralize
import numpy as np
from src.constants import PREPROCCESSED_DATA_DIR, FULL_SENTENCE_DATA_DIR



def transform_answer(tagged_question,answer,question,data):
  new_answer=""
  tagged_question = [(str.lower(tag[0]),tag[1]) for tag in tagged_question[:-1]]

  #What
  if tagged_question[0][1] == 'WP':
    #print('here12')
    try:
      if tagged_question[1][0] == 'does':

        are = False
        if ',' in answer:
          are = True
          answer = answer.replace(',', " and" )

        new_answer = ""
        regex = r"^VB"
        idx=1

        for tag in tagged_question[idx:]:
            if re.match(regex, tag[1]):
              for tag in tagged_question[idx+1:-1]:
                new_answer+=tag[0]+" "
                idx+=1 
              break   
          
        new_answer+= conjugate(tagged_question[idx+1][0],tense=PRESENT) + " "
        if are== False:
          new_answer+= 'is ' + answer
        else:
          new_answer+= 'are ' + answer  


      else:  
        VP=""
        if answer == 'None':
          answer = 'No'

        are = False
        if ',' in answer:
          are = True  
          answer = answer.replace(',', " and" )

        new_answer = answer+ " "
        regex_v = r"^VB"
        regex_n = r"^NN"
        idx=1

        
        if are == True:
          new_answer+= 'are ' 
          first = True
          for tag in tagged_question[2:]:
            if tag[1] == 'NN':
              if first:
                new_answer+= pluralize(tag[0])+" "
                idx+=1   
                break
            else:
              new_answer+=tag[0]+" "
              idx+=1    

          for tag in tagged_question[idx+1:]:  
            new_answer+=tag[0]+" "

        else:
          idx=1
          for tag in tagged_question[1:]:
            if re.match(regex_v, tag[1]) or re.match(regex_n, tag[1]):
              for tag in tagged_question[idx:]:
                VP+=tag[0]+" "
              break   
            idx+=1 
          new_answer+=VP

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}"
#---------------------------------------------------#

 #IN
  elif tagged_question[0][1] == 'IN':
    #print('here11')
    try:
      NP=""
      VP=""
      new_answer = ""
      regex_n = r"^DT"
      regex_v = r"^VB"
      idx=0

      for tag in tagged_question:
        if re.match(regex_n, tag[1]):
            NP+=tag[0]+" "
            break
        idx+=1

      NP += tagged_question[idx+1][0] + ' '

      for tag in tagged_question:
        if re.match(regex_v, tag[1]):
            VP+=tag[0]+" "
            break

      new_answer+= NP + VP + tagged_question[-1][0] + ' in ' + answer
     

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}" 
#---------------------------------------------------#

  #How many
  elif tagged_question[0][1] == 'WRB' and tagged_question[1][1] == 'JJ' and tagged_question[0][0] == 'how' and tagged_question[1][0] == 'many':
    #print('here10')
    try:
      NP=""
      VP=""
      new_answer = ""
      regex_n = r"^NN"
      regex_v = r"^VB"
      idx=0

      for tag in tagged_question:
        if re.match(regex_n, tag[1]):
            NP+=tag[0]+" "
            break

      for tag in tagged_question:
        if not re.match(regex_v, tag[1]):
          idx+=1
        else:
          VP+=tag[0]+" "
          idx+=1
          break

      new_answer+=tagged_question[idx][0] + " " + VP + answer + " " + NP 
      
      for tag in tagged_question[idx+1:]:
        new_answer+=tag[0] + " "

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}" 
#---------------------------------------------------#

#How many
  elif tagged_question[0][1] == 'WRB' and tagged_question[1][1] == 'JJ' and tagged_question[0][0] == 'how' and tagged_question[1][0] == 'much':
    #print('here9')
    try:
      new_answer = answer + ' '

      for tag in tagged_question[2:]:
        new_answer+=tag[0] + " "

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}" 
#---------------------------------------------------#

#How 
  elif tagged_question[0][1] == 'WRB' and tagged_question[1][1] == 'TO':
    #print('here8')
    try:
      NP=""
      VP=""

      if ',' in answer:
        answer = answer.replace(',', " and" )
      new_answer = answer + " "
      for tag in tagged_question[1:]:
        new_answer+=tag[0] + " "

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}" 
#---------------------------------------------------#

  #Where
  elif tagged_question[0][1] == 'WRB' and tagged_question[0][0] == 'where':
    #print('here7')
    try:
      are = False
      in_ = False
      if ',' in answer:
        are = True

      if tagged_question[1][0] == 'are':
        are = True

      NP=""
      new_answer = ""
      regex = r"^VB"
      idx=1
      for tag in tagged_question[1:]:
          if re.match(regex, tag[1]):
            idx+=1
            continue
          else: 
            if tag[1] != 'NN':
              NP+=tag[0]+" "
              idx+=1
              break
    
      new_answer+=NP 
      
      if are == False:
        new_answer+= tagged_question[idx][0] + ' is '
      else:
          new_answer+= pluralize(tagged_question[idx][0]) + ' are '

      try:
        new_answer+=tagged_question[idx+1][0]+ ' ' 
      except:
        new_answer=new_answer

      
      if answer != 'Not seen':

        for tag in tagged_question[1:]:
          if tag[0] == 'in':
            in_ = True
          if tag[0] == 'image' or tag[0] == 'picture':  
            in_ = False
       
        words = new_answer.split()
        for word in words:
          if word =='in': 
            in_ = True 
        
        if in_ : 
          new_answer+= 'the '+answer
        if not in_ : 
          new_answer+= 'in the '+answer
      else:
          new_answer+= answer

      new_answer=new_answer.strip()
      new_answer+='.'
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}"  

#---------------------------------------------------#

  #Which + Is
  elif tagged_question[0][1] == 'WDT':
    #print('here6')
    try:
      new_answer = answer+ " "
      for tag in tagged_question[1:]:
        if tag[1] != ',':
          new_answer +=tag[0]+" "
        else: 
          break
      new_answer=new_answer.strip()
      new_answer+='.' 
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}"  

  #Which  2

  elif tagged_question[0][1] == 'NNP' and tagged_question[0][0] == 'which':
    #print('here5')
    try:
      are = False
      if ',' in answer:
        are = True

      if str.lower(answer) not in question:
        if answer != 'None':
          new_answer = answer+ " are "
      else:
          new_answer = answer+ " "
          if are == False:
            new_answer += 'is '
          else:
            new_answer += 'are '

      for tag in tagged_question[3:]:
        if tag[1] != ',':
         new_answer +=tag[0]+" "
        else: 
         break
      new_answer=new_answer.strip()
      new_answer+='.' 
      new_answer = new_answer.capitalize()

    except:
      new_answer=f"question probably wrong: {data}"   


#---------------------------------------------------#

  #Yes/No
  elif tagged_question[0][1] == 'VBZ' or tagged_question[0][1] == 'VBP' or tagged_question[0][1] == 'VB':
    #print('here4')
    try:
      #any
      if tagged_question[1][0] == 'any' and answer == "Yes":
        new_answer = answer+ ","+" "
        for tag in tagged_question[3:]:
          new_answer+=tag[0] + " "

        new_answer=new_answer.strip()
        new_answer+='.'    

      #Yes
      elif answer == "Yes":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        regex = r"^NN"
        for tag in tagged_question[1:]:
          if not re.match(regex, tag[1]):
            NP+=tag[0]+" "
            idx+=1
            break

        NP+=tagged_question[idx][0]+" "  
        new_answer+=NP
        verb=""
        
        if tagged_question[0][0] == 'is':
          verb = "is "+ tagged_question[idx+1][0]

        elif tagged_question[0][0] == 'are':
          verb = "are "+ tagged_question[idx+1][0]

        elif tagged_question[0][0] == 'does' or tagged_question[0][0] == 'do':
          verb = conjugate(tagged_question[idx+1][0],tense=PRESENT)

        new_answer+=verb+" "
        for word in tagged_question[idx+2:]:
          if word[0] == 'any':
            new_answer+='some '
            continue
          new_answer+=word[0] + " "
        new_answer=new_answer.strip()
        new_answer+='.'

    #No
      elif answer == "No":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        for tag in tagged_question[1:]:
          if tag[1] != 'NN':
            NP+=tag[0]+" "
            idx+=1
            break

        NP+=tagged_question[idx][0]+" "  
        new_answer+=NP
        new_answer+=str.lower(tagged_question[0][0])+" not " + tagged_question[idx+1][0] + " "

        for word in tagged_question[idx+2:]:
          new_answer+=word[0] + " "
        new_answer=new_answer.strip()
        new_answer+='.' 
      

     #not yes/no 
      else: 
        new_answer=""
        idx = 0
        NP=""
        for tag in tagged_question[1:]:
          if tag[0] == 'or':
            break
          else:
            idx+=1 

        for tag in tagged_question[1:idx]:   
          NP+= tag[0] + " "

        if tagged_question[0][0] == 'is':
          new_answer += NP + 'is '+ answer + '.'


      new_answer = new_answer.capitalize()  

    except:
      new_answer=f"question probably wrong: {data}"    

#---------------------------------------------------#

  #Yes/No can
  elif tagged_question[0][1] == 'MD' :
    #print('here3')
    try:
      #Yes
      if answer == "Yes":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        regex = r"^NN"
        for tag in tagged_question[1:]:
          if not re.match(regex, tag[1]):
            NP+=tag[0]+" "
            idx+=1
            break

        NP+=tagged_question[idx][0]+" "  
        new_answer+=NP
        verb=""
        
        if tagged_question[0][0] == 'is':
          verb = "is "+ tagged_question[idx+1][0]

        elif tagged_question[0][0] == 'are':
          verb = "are "+ tagged_question[idx+1][0]

        elif tagged_question[0][0] == 'does':
          verb = conjugate(tagged_question[idx+1][0],tense=PRESENT)

        new_answer+=verb+" "
        for word in tagged_question[idx+2:]:
          new_answer+=word[0] + " "
        new_answer=new_answer.strip()
        new_answer+='.'

    #No
      elif answer == "No":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        regex = r"^NN"
        for tag in tagged_question[1:]:
          if not re.match(regex, tag[1]):
            NP+=tag[0]+" "
            idx+=1
            break

        NP+=tagged_question[idx][0]+" "  
        new_answer+=NP
        new_answer+=str.lower(tagged_question[0][0])+" not " + tagged_question[idx+1][0] + " "
        
        for word in tagged_question[idx+2:]:
          new_answer+=word[0] + " "
        new_answer=new_answer.strip()
        new_answer+='.' 
      new_answer = new_answer.capitalize()  

    except:
      new_answer=f"question probably wrong: {data}"    

      #----------------------------#

  #Yes/No 2

  elif tagged_question[0][1] == 'NNP':
    #print('here2')
    try:
      #Yes
      if answer == "Yes":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        for tag in tagged_question[1:]:
          if tag[1] == 'DT' or tag[1] == 'EX':
            NP+=tag[0]+" "
            idx+=1
            break

        NP+=tagged_question[0][0]+" "  
        new_answer+=NP

        for word in tagged_question[idx:]:
          new_answer+=word[0]+ ' '

        new_answer=new_answer.strip()
        new_answer+='.'
        

      #No
      elif answer == "No":
        NP=""
        new_answer = answer+ ","+" "
        idx=1
        for tag in tagged_question[1:]:
          if tag[1] == 'DT' or tag[1] == 'EX':
            NP+=tag[0]+" "
            idx+=1
            break
        NP+=tagged_question[0][0]+" no "
        new_answer+=NP

        for word in tagged_question[idx:]:
          new_answer+=word[0]+ ' '

        new_answer=new_answer.strip()
        new_answer+='.'  
      new_answer = new_answer.capitalize()
    except:
      new_answer=f"question probably wrong: {data}"    

#---------------------------------------------------#

  #Which + Noun
  elif tagged_question[0][1] == 'JJ':
    #print('here1')
    try:

        if tagged_question[2][0] == 'of':
          
          new_answer=""
          idx_1=1
          for tag in tagged_question[1:]:
            if tag[1] != 'VBZ':
              idx_1+=1
            else:
              break  
        
          idx_2 = idx_1 + 1
          for tag in tagged_question[idx_1:]:
            if tag[1] != 'DT':
              idx_2+=1
            else:
              break  
          
          try:
            for tag in tagged_question[idx_1+1:idx_2+1]:
              new_answer+= tag[0] + " "
            

            VP = conjugate(tagged_question[idx_2+1][0],tense=PRESENT) + " "

            new_answer+=VP
            for tag in tagged_question[idx_2+2:]:
              new_answer+= tag[0] + " "

            new_answer+=answer

          except:
            new_answer = answer + " "
            for tag in tagged_question[1:]:
              if tag[0] ==',':
                break
              else:
                new_answer+= tag[0] + " "



        else:
          new_answer=answer + " "
          idx_1=1
          regex_1 = r"^NN"
          regex_2 = r"^VBP"
          idx=1
          VP=""
          for tag in tagged_question[1:]:
            if re.match(regex_1, tag[1]):
              if answer == 'Both':
                noun = pluralize(tag[0])
                new_answer+=noun+" "  
                for tag in tagged_question[idx+1:]:
                  if tag[0] == 'is' and answer == 'Both':
                    new_answer+='are '
                    continue
                  if tag[0] ==',':
                    break
                  else:
                    new_answer+= tag[0] + " "
                break

              else:
                for tag in tagged_question[idx:]:
                  if tag[0] == 'is' and answer == 'Both':
                    new_answer+='are '
                    continue
                  if tag[0] ==',':
                    break
                  else:
                    new_answer+= tag[0] + " "
                break  
            
            elif re.match(regex_2, tag[1]):
              VP = conjugate(tag[0],tense=PRESENT) + " "
              if answer == 'Both':
                new_answer+='are '
              else:
                new_answer+=VP
              for tag in tagged_question[idx+1:]:
                new_answer+=tag[0]+" "  
              break
            idx+=1  

        new_answer=new_answer.strip()
        new_answer+='.'
        new_answer = new_answer.capitalize()     

    
    except:
      new_answer=f"question probably wrong: {data}"    

#-------------
  else:
    new_answer = f"question probably wrong: {data}"    

  return new_answer


def generate_full_sentences(PREPROCCESSED_DATA_DIR, FULL_SENTENCE_DATA_DIR):
    train_dir = PREPROCCESSED_DATA_DIR + "train_en.json"
    val_dir = PREPROCCESSED_DATA_DIR + "val_en.json"
    test_dir = PREPROCCESSED_DATA_DIR + "test_en.json"

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()

    f = open(val_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()

    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()

    #train
    for data in train_data:
        question=data['question']
        answer=data['answer']
        text = nltk.word_tokenize(question)
        tagged_question = nltk.pos_tag(text)
        data['fs_answer'] = transform_answer(tagged_question,answer,question, data)

    j = json.dumps(train_data)      
    with open(FULL_SENTENCE_DATA_DIR+'/train_full_sentence.json', 'a') as fp:
        fp.write(j)
    train_data =[]  

    #val
    for data in val_data:
        question=data['question']
        answer=data['answer']
        text = nltk.word_tokenize(question)
        tagged_question = nltk.pos_tag(text)
        data['fs_answer'] = transform_answer(tagged_question,answer,question, data)

    j = json.dumps(val_data)      
    with open(FULL_SENTENCE_DATA_DIR+'/val_full_sentence.json', 'a') as fp:
        fp.write(j)
    val_data =[] 

    #test
    for data in test_data:
        question=data['question']
        answer=data['answer']
        text = nltk.word_tokenize(question)
        tagged_question = nltk.pos_tag(text)
        data['fs_answer'] = transform_answer(tagged_question,answer,question, data)

    j = json.dumps(test_data)      
    with open(FULL_SENTENCE_DATA_DIR+'/test_full_sentence.json', 'a') as fp:
        fp.write(j)
    test_data =[]   