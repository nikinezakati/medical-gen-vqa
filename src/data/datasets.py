import pickle
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import json
class GenVQADataset(Dataset):
    def __init__(self, tokenizer, annotations, questions, img_dir, batch_size=32):
        with open(annotations, encoding="utf8") as f:
            self.annotations = json.load(f)
        with open(questions, encoding="utf8") as f:
            self.questions = json.load(f)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __getitem__(self, idx):
        dataum = self.annotations[idx]
        q = self.questions[str(dataum['qid'])]
        img_path = os.path.join(self.img_dir, f"{dataum['img_id']}.pkl")
        
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        # extract sentence data
        tokenized_sentence = self.tokenizer(q['question'])
        input_ids = tokenized_sentence['input_ids']
        attention_mask = tokenized_sentence['attention_mask']
        # extract image data
        visual_feats = img['features']
        padd_num = 2048 - len(visual_feats)
        padd = torch.zeros(padd_num)
        visual_feats_padded = torch.cat((visual_feats,padd))
        visual_feats_padded = torch.unsqueeze(visual_feats_padded, 0)
        padd = torch.zeros(35,2048)
        visual_feats = torch.cat((visual_feats_padded,padd),0)

        boxes = img['boxes']
        padd_num = 36 - len(boxes)
        padd = torch.zeros(padd_num,4)
        boxes_padded = torch.cat((boxes,padd),0)

        img_h, img_w = img['img_h'], img['img_w']
        boxes = boxes_padded.clone().detach()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        visual_pos = boxes

        # labels
        if 'fs_answer' in dataum.keys():
          a_text = dataum['fs_answer']
          tokenized_sentence = self.tokenizer(a_text)
          label_tokenized = tokenized_sentence['input_ids']
          # label_masks = tokenized_sentence['attention_mask']
          return input_ids, visual_feats, visual_pos, attention_mask, label_tokenized
        
        return input_ids, visual_feats, visual_pos, attention_mask, None
    
    def __len__(self):
        return len(self.annotations)

def pad_batched_sequence(batch):
    
    input_ids = [torch.tensor(item[0]) for item in batch]
    visual_feats =  [torch.tensor(item[1]) for item in batch]
    visual_pos =  [torch.tensor(item[2]) for item in batch]
    attention_mask =  [torch.tensor(item[3]) for item in batch]
    
    input_ids = pad_sequence(input_ids, padding_value=0, batch_first=True)
    attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
    label_tokenized = None
    # label_masks = None
    
    if batch[0][4]:
        #Ignore statrt idx
        label_tokenized = [torch.tensor(item[4]) for item in batch]
        # label_tokenized = [torch.tensor(item[4][1:]) for item in batch]
        label_tokenized = pad_sequence(label_tokenized, batch_first=False, padding_value=0).cuda()
    
    return input_ids.cuda(), torch.stack(visual_feats, dim=0).cuda(), torch.stack(visual_pos, dim=0).cuda(), attention_mask.cuda(), label_tokenized
