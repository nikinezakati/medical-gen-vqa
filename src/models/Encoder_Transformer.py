import os
import random
import torch
from torch import nn

import torch.nn.functional as F
from transformers import LxmertModel, LxmertTokenizer, VisualBertModel, BertTokenizer

from src.utils import PositionalEncoder

class Encoder_Transformer(nn.Module):
    def __init__(self, encoder_type, nheads, decoder_layers, hidden_size, freeze_encoder=True):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'lxmert':
            self.encoder = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
            self.encoder.config.output_hidden_states = True
            self.Tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
            
        elif encoder_type == 'visualbert':
            self.encoder = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            self.Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        #freeze LXMERT
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        
        # This standard decoder layer is based on the paper “Attention Is All You Need”.
        transformer_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nheads)
        self.Decoder = nn.TransformerDecoder(transformer_layer, num_layers=decoder_layers)

        self.pe = PositionalEncoder(hidden_size, dropout=0.1,max_len=200)
        
        self.embedding_layer = self.encoder.embeddings.word_embeddings
        self.output_size = self.Tokenizer.vocab_size
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.hidden_size = hidden_size
        self.PADDING_VALUE = 0
        self.START_TOKEN = 101
        self.SEP_TOKEN = 102 
        #Linear layer to output vocabulary size
        self.Linear = nn.Linear(hidden_size, self.Tokenizer.vocab_size)
        
        self.name = f"{encoder_type}_{nheads}heads_{decoder_layers}_transformer"
        print(self.name)
    
    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, answer_tokenized=None, teacher_force_ratio=0.5, max_seq_len=50):
        """
            Train phase forward propagation
        """
        batch_size = input_ids.shape[0]
        max_seq_len = max_seq_len if answer_tokenized is None else answer_tokenized.shape[0]

        # shift right
        answer_tokenized = answer_tokenized[:-1,:] if answer_tokenized is not None else answer_tokenized
        
        # encode question and image with lxmert
        if self.encoder_type == 'lxmert':
            kwargs = {"input_ids" : input_ids,
                      "visual_feats": visual_feats,
                      "visual_pos" : visual_pos,
                      "attention_mask": attention_mask}
            output = self.encoder(**kwargs)
            encoder_output  = output.language_hidden_states[-1].permute(1, 0, 2)
            # encoder_output shape: (seq_len, N, hidden_size) to send it to
            
            # memory masks to consider padding values in source sentence (questions)
            memory_key_padding_mask = (input_ids == self.PADDING_VALUE)
        
        elif self.encoder_type == 'visualbert':
            kwargs = {"input_ids" : input_ids,
                      "attention_mask": attention_mask,
                      "visual_embeds": visual_feats,
                      "output_hidden_states":True}
            output = self.encoder(**kwargs)
            encoder_output = output.hidden_states[-1].permute(1,0,2)
            # encoder_output shape: (sequence_length, batch_size, hidden_size)
            
            # memory masks to consider padding values in source sentence (questions)
            memory_key_padding_mask = (input_ids == self.PADDING_VALUE)
            memory_key_padding_mask = F.pad(input=memory_key_padding_mask, pad=(0, visual_feats.shape[1], 0, 0), mode='constant', value=0)
            # (batch_size, text_seq_length+image_seq_length)
        
        # if answer_tokenized is not None and random.random() < teacher_force_ratio:
        if answer_tokenized is not None:
            tgt_len = answer_tokenized.shape[0]

            answer_embeddings = self.embedding_layer(answer_tokenized)
            # answer embeddings shape: (seq_len, N, embedding_size)
            # embedding_size is 768 in LXMERT
            positions = self.pe(answer_embeddings)
            
            # target masks to consider padding values in target embeddings (answers)
            tgt_key_padding_mask = (answer_tokenized.permute(1, 0) == self.PADDING_VALUE)

            # target attention masks to avoid future tokens in our predictions
            # Adapted from PyTorch source code:
            # https://github.com/pytorch/pytorch/blob/176174a68ba2d36b9a5aaef0943421682ecc66d4/torch/nn/modules/transformer.py#L130
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()        
            

            # decode sentence and encoder output to generate answer
            output = self.Decoder(positions, 
                                encoder_output,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask, 
                                memory_key_padding_mask = memory_key_padding_mask)
            #output shape: (tgt_seq_len, N, hidden_size)

            output = self.Linear(output)
            #output shape: (tgt_seq_len, N, vocab_size)
        
        else:
            # generatoin phase
            x = torch.tensor([[self.START_TOKEN] * batch_size]).cuda()
            # x shape: (1, N)
            target_vocab_size = self.Tokenizer.vocab_size

            outputs = torch.zeros(max_seq_len, batch_size, target_vocab_size).cuda()
            # outputs[0,:,self.START_TOKEN] = 1
            
            for i in range(0,max_seq_len):
                tgt_len = x.shape[0]
                answer_embeddings = self.embedding_layer(x)
                positions = self.pe(answer_embeddings)
                tgt_key_padding_mask = (x.permute(1, 0) == self.PADDING_VALUE)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()
                output = self.Decoder( 
                                positions, 
                                encoder_output,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask, 
                                memory_key_padding_mask = memory_key_padding_mask) 
                #output shape: (tgt_seq_len, N, hidden_size)
                
                # chose the last word of sequence Based on CodeXGLUE project
                # https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-refinement/code/model.py
                output = output[-1, :, :].unsqueeze(0)
                #output shape (1, N, hidden_size)
                

                output = self.Linear(output)
                #output shape: (1, N, vocab_size)
                
                outputs[i] = output
                
                #consider best guesses in a greeedy form! Better to implement with beam search
                output = torch.argmax(output, dim = -1)
                #output shape: (1, N)

                #concat new generated answer to x.
                x = torch.cat([x, output], dim=0)
                # x shape: (i + 1, N)
            
            output = outputs

        return output

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
