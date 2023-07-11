import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, LSTM


class BahdanauRNN(torch.nn.Module):
    def __init__(self, embedding, rnn_type="lstm", hidden_size=768, output_size=768, prob=0.5):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        
        self.dropout = nn.Dropout(prob)
        
        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    
        if self.rnn_type=="lstm":
            self.rnn = LSTM(input_size=hidden_size*2, hidden_size=self.hidden_size)
        elif self.rnn_type=="gru":
            self.rnn = GRU(input_size=hidden_size*2, hidden_size=self.hidden_size)
            
        self.Linear = nn.Linear(self.hidden_size, output_size)
        
        
    def forward(self, inputs, hidden, encoder_states):
        """
            inputs shape: (batch_size) 
            
            hidden shape: (1, batch_size, hidden_size)
            
            encoder_states shape: (sequence_length, batch_size, hidden_size)
        """
        # we want input to be (1, batch_size), seq_length
        # is 1 here because we are sending in a single word and not a sentence.
        inputs = inputs.unsqueeze(0)
        embedding = self.dropout(self.embedding(inputs))
        # embedding shape: (1, batch_size, embedding_size)
        
        h = hidden if self.rnn_type == 'gru' else hidden[0]
        
        # Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(h)+self.fc_encoder(encoder_states))
        # x shape:(sequence_length, batch_size, hidden_size)
        
        batch_size = encoder_states.shape[1]
        alignment_scores = torch.bmm(x.permute(1,0,2), self.weight.unsqueeze(2).repeat(batch_size,1,1))
        # alignment_scores shape:(batch_size, sequence_length, 1)

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)
        # attn_weights shape:(batch_size, sequence_length, 1)

        
        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.permute(0,2,1), encoder_states.permute(1,0,2))
        # context_vector shape: (batch_size, 1, hidden_state)

        # Concatenating context vector with embedded input word
        input_rnn = torch.cat((embedding, context_vector.permute(1,0,2)), 2)
        # input_rnn shape: (1, batch_size, 2*hidden_size)

        
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.rnn(input_rnn, hidden)
        # output shape: (1, batch_size, hidden_size)

        predictions = self.Linear(output)

        # predictions shape: (1, batch_size, length_target_vocabulary) to send it to
        # loss function we want it to be (batch_size, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, attn_weights
