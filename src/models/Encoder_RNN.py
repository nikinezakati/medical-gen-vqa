import os
import random

import torch
from transformers import LxmertModel, LxmertTokenizer, VisualBertModel, BertTokenizer

from src.models.RNN import RNNModel


class Encoder_RNN(torch.nn.Module):
    def __init__(self, encoder_type='lxmert', rnn_type = "lstm", num_layers=1, bidirectional=False, prob=0.5, freeze_encoder=True):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'lxmert':
            self.encoder = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
            self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        elif encoder_type == 'visualbert':
            self.encoder = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            self.Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        #freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        
        self.embedding_layer = self.encoder.embeddings.word_embeddings

        self.rnn = RNNModel(embedding=self.embedding_layer,
                            rnn_type=rnn_type,  
                            output_size=self.Tokenizer.vocab_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            prob=prob)
        
        self.name = f"{encoder_type}_{rnn_type}_{num_layers}"
        self.name = f"{self.name}_bidirectional" if bidirectional else self.name
        
        self.start_token = 101 # <cls>
        self.end_token = 102 # <sep>
        
        self.D = 2 if bidirectional==True else 1
        
        print(self.name)
    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized = None, teacher_force_ratio=0.5, max_sequence_length=50):
        """
            Train phase forward propagation
        """
        
        batch_size = input_ids.shape[0]
        target_len = max_sequence_length if answer_tokenized is None else answer_tokenized.shape[0]
        target_vocab_size = self.Tokenizer.vocab_size
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).cuda()
        
        #encoder
        if self.encoder_type == 'lxmert':
            kwargs = {"input_ids" : input_ids,
                    "visual_feats": visual_feats,
                    "visual_pos" : visual_pos,
                    "attention_mask": attention_mask}
            output = self.encoder(**kwargs)
            encoder_output = output.pooled_output
            
        elif self.encoder_type == 'visualbert':
            kwargs = {"input_ids" : input_ids,
                      "attention_mask": attention_mask,
                      "visual_embeds": visual_feats}
            output = self.encoder(**kwargs)
            encoder_output = output.pooler_output
            
        # encoder_output shape: (N, hidden_size) to send it to Decoder as hidden,
        # we want it to be (D*num_layers, N, hidden_size) so we're just gonna expand it.
        
        h = encoder_output.expand(self.D*self.rnn.num_layers, -1, -1)
        # h shape: (D*num_layers, N, hidden_size)
        
        if self.rnn.rnn_type == 'lstm':
            c = torch.zeros(*h.shape).cuda()
            hidden = (h.contiguous(),c.contiguous())
            
        elif self.rnn.rnn_type == 'gru':
            hidden = h.contiguous()
            
        # Send <cls> token to decoder
        x =  torch.tensor([self.start_token]*batch_size).cuda()
        
        for t in range(0, target_len):
            output, hidden = self.rnn(x, hidden)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to.
            x = answer_tokenized[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
