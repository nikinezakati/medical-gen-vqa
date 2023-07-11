import json
import sys
from src.constants import FULL_SENTENCE_DATA_DIR, RAW_DATA_DIR, CLASSIFIED_DATA_DIR

def classify_data(data_output_dir=CLASSIFIED_DATA_DIR):
    dir = FULL_SENTENCE_DATA_DIR

    train_dir = dir + 'train_full_sentence.json'
    validate_dir = dir + 'val_full_sentence.json'
    test_dir = dir + 'test_full_sentence.json'

    f = open(train_dir, encoding="utf8")
    train_data = json.load(f)
    f.close()
    Plane=[]
    Quality=[]
    Modality=[]
    Position=[]
    Organ=[]
    k_g=[]
    Abnormal=[]
    Color=[]
    Shape=[]
    Size=[]

    for tr in train_data:
        if tr['content_type'] == 'Plane':
           Plane.append(tr)   

        elif tr['content_type'] == 'Quality':
           Quality.append(tr)    

        elif tr['content_type'] == 'Modality':
           Modality.append(tr)   

        elif tr['content_type'] == 'Position':
           Position.append(tr)     

        elif tr['content_type'] == 'Organ':
           Organ.append(tr)     

        elif tr['content_type'] == 'KG':
           k_g.append(tr) 

        elif tr['content_type'] == 'Abnormal':
           Abnormal.append(tr) 

        elif tr['content_type'] == 'Color':
           Color.append(tr)   

        elif tr['content_type'] == 'Shape':
           Shape.append(tr)  

        elif tr['content_type'] == 'Size':
           Size.append(tr)                                

    j = json.dumps(Plane)        
    with open(CLASSIFIED_DATA_DIR+'/Plane.json', 'a') as fp:
        fp.write(j)
    Plane =[]

    j = json.dumps(Quality)        
    with open(CLASSIFIED_DATA_DIR+'/Quality.json', 'a') as fp:
        fp.write(j)
    Quality =[]

    j = json.dumps(Modality)        
    with open(CLASSIFIED_DATA_DIR+'/Modality.json', 'a') as fp:
        fp.write(j)
    Modality =[]

    j = json.dumps(Position)        
    with open(CLASSIFIED_DATA_DIR+'/Position.json', 'a') as fp:
        fp.write(j)
    Position =[]

    j = json.dumps(Organ)        
    with open(CLASSIFIED_DATA_DIR+'/Organ.json', 'a') as fp:
        fp.write(j)
    Organ =[]

    j = json.dumps(k_g)        
    with open(CLASSIFIED_DATA_DIR+'/KG.json', 'a') as fp:
        fp.write(j)
    k_g =[]

    j = json.dumps(Abnormal)        
    with open(CLASSIFIED_DATA_DIR+'/Abnormal.json', 'a') as fp:
        fp.write(j)
    Abnormal =[]

    j = json.dumps(Color)        
    with open(CLASSIFIED_DATA_DIR+'/Color.json', 'a') as fp:
        fp.write(j)
    Color =[]

    j = json.dumps(Shape)        
    with open(CLASSIFIED_DATA_DIR+'/Shape.json', 'a') as fp:
        fp.write(j)
    Shape =[]

    j = json.dumps(Size)        
    with open(CLASSIFIED_DATA_DIR+'/Size.json', 'a') as fp:
        fp.write(j)
    Size =[]



    f = open(validate_dir, encoding="utf8")
    val_data = json.load(f)
    f.close()
    Plane=[]
    Quality=[]
    Modality=[]
    Position=[]
    Organ=[]
    k_g=[]
    Abnormal=[]
    Color=[]
    Shape=[]
    Size=[]

    for tr in val_data:
        if tr['content_type'] == 'Plane':
           Plane.append(tr)   

        elif tr['content_type'] == 'Quality':
           Quality.append(tr)    

        elif tr['content_type'] == 'Modality':
           Modality.append(tr)   

        elif tr['content_type'] == 'Position':
           Position.append(tr)     

        elif tr['content_type'] == 'Organ':
           Organ.append(tr)     

        elif tr['content_type'] == 'KG':
           k_g.append(tr) 

        elif tr['content_type'] == 'Abnormal':
           Abnormal.append(tr) 

        elif tr['content_type'] == 'Color':
           Color.append(tr)   

        elif tr['content_type'] == 'Shape':
           Shape.append(tr)  

        elif tr['content_type'] == 'Size':
           Size.append(tr)                                

    j = json.dumps(Plane)        
    with open(CLASSIFIED_DATA_DIR+'/Plane.json', 'a') as fp:
        fp.write(j)
    Plane =[]

    j = json.dumps(Quality)        
    with open(CLASSIFIED_DATA_DIR+'/Quality.json', 'a') as fp:
        fp.write(j)
    Quality =[]

    j = json.dumps(Modality)        
    with open(CLASSIFIED_DATA_DIR+'/Modality.json', 'a') as fp:
        fp.write(j)
    Modality =[]

    j = json.dumps(Position)        
    with open(CLASSIFIED_DATA_DIR+'/Position.json', 'a') as fp:
        fp.write(j)
    Position =[]

    j = json.dumps(Organ)        
    with open(CLASSIFIED_DATA_DIR+'/Organ.json', 'a') as fp:
        fp.write(j)
    Organ =[]

    j = json.dumps(k_g)        
    with open(CLASSIFIED_DATA_DIR+'/KG.json', 'a') as fp:
        fp.write(j)
    k_g =[]

    j = json.dumps(Abnormal)        
    with open(CLASSIFIED_DATA_DIR+'/Abnormal.json', 'a') as fp:
        fp.write(j)
    Abnormal =[]

    j = json.dumps(Color)        
    with open(CLASSIFIED_DATA_DIR+'/Color.json', 'a') as fp:
        fp.write(j)
    Color =[]

    j = json.dumps(Shape)        
    with open(CLASSIFIED_DATA_DIR+'/Shape.json', 'a') as fp:
        fp.write(j)
    Shape =[]

    j = json.dumps(Size)        
    with open(CLASSIFIED_DATA_DIR+'/Size.json', 'a') as fp:
        fp.write(j)
    Size =[]



    f = open(test_dir, encoding="utf8")
    test_data = json.load(f)
    f.close()
    Plane=[]
    Quality=[]
    Modality=[]
    Position=[]
    Organ=[]
    k_g=[]
    Abnormal=[]
    Color=[]
    Shape=[]
    Size=[]

    for tr in test_data:
        if tr['content_type'] == 'Plane':
           Plane.append(tr)   

        elif tr['content_type'] == 'Quality':
           Quality.append(tr)    

        elif tr['content_type'] == 'Modality':
           Modality.append(tr)   

        elif tr['content_type'] == 'Position':
           Position.append(tr)     

        elif tr['content_type'] == 'Organ':
           Organ.append(tr)     

        elif tr['content_type'] == 'KG':
           k_g.append(tr) 

        elif tr['content_type'] == 'Abnormal':
           Abnormal.append(tr) 

        elif tr['content_type'] == 'Color':
           Color.append(tr)   

        elif tr['content_type'] == 'Shape':
           Shape.append(tr)  

        elif tr['content_type'] == 'Size':
           Size.append(tr)                                

    j = json.dumps(Plane)        
    with open(CLASSIFIED_DATA_DIR+'/Plane.json', 'a') as fp:
        fp.write(j)
    Plane =[]

    j = json.dumps(Quality)        
    with open(CLASSIFIED_DATA_DIR+'/Quality.json', 'a') as fp:
        fp.write(j)
    Quality =[]

    j = json.dumps(Modality)        
    with open(CLASSIFIED_DATA_DIR+'/Modality.json', 'a') as fp:
        fp.write(j)
    Modality =[]

    j = json.dumps(Position)        
    with open(CLASSIFIED_DATA_DIR+'/Position.json', 'a') as fp:
        fp.write(j)
    Position =[]

    j = json.dumps(Organ)        
    with open(CLASSIFIED_DATA_DIR+'/Organ.json', 'a') as fp:
        fp.write(j)
    Organ =[]

    j = json.dumps(k_g)        
    with open(CLASSIFIED_DATA_DIR+'/KG.json', 'a') as fp:
        fp.write(j)
    k_g =[]

    j = json.dumps(Abnormal)        
    with open(CLASSIFIED_DATA_DIR+'/Abnormal.json', 'a') as fp:
        fp.write(j)
    Abnormal =[]

    j = json.dumps(Color)        
    with open(CLASSIFIED_DATA_DIR+'/Color.json', 'a') as fp:
        fp.write(j)
    Color =[]

    j = json.dumps(Shape)        
    with open(CLASSIFIED_DATA_DIR+'/Shape.json', 'a') as fp:
        fp.write(j)
    Shape =[]

    j = json.dumps(Size)        
    with open(CLASSIFIED_DATA_DIR+'/Size.json', 'a') as fp:
        fp.write(j)
    Size =[]    




if __name__ == "__main__":
    try:
        dir = sys.argv[1]
    except:
        dir = RAW_DATA_DIR
    classify_data(dir)        

