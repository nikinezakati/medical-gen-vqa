import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, LSTM


class LuongRNN(torch.nn.Module):
    def __init__(self, embedding, rnn_type="lstm", attn_method="dot", hidden_size=768, output_size=768, prob=0.5):
        super().__init__()
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_type = rnn_type
        
        self.attention = Attention(hidden_size,attn_method)
        
        self.dropout = nn.Dropout(prob)
        
        if self.rnn_type=="lstm":
            self.rnn = LSTM(input_size=hidden_size, hidden_size=self.hidden_size)
        elif self.rnn_type=="gru":
            self.rnn = GRU(input_size=hidden_size, hidden_size=self.hidden_size)
            
        self.Linear = nn.Linear(self.hidden_size*2, output_size)
        
        
    def forward(self, inputs, hidden, encoder_states):
        """
            input shape: (batch_size) 
            
            hidden shape: (1, batch_size, hidden_size)
            
            encoder_states shape: (sequence_length, batch_size, hidden_size)
        """
        # we want input to be (1, batch_size), seq_length
        # is 1 here because we are sending in a single word and not a sentence.
        inputs = inputs.unsqueeze(0)
        embedding = self.dropout(self.embedding(inputs))
        # embedding shape: (1, batch_size, embedding_size)
        
        # Passing previous output word (embedded) and hidden state into LSTM cell
        rnn_out, hidden = self.rnn(embedding, hidden)
        # rnn_out shape: (1, batch_size, hidden_size)
        
        h = hidden if self.rnn_type == 'gru' else hidden[0]
        # Calculating Alignment Scores
        alignment_scores = self.attention(h, encoder_states)
        # alignment_scores shape: (batch_size, sequence_lenght, 1)
        
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)
        # attn_weights shape: (batch_size, sequence_lenght, 1)
        
        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.permute(0,2,1), encoder_states.permute(1,0,2))
        # context_vector shape: (batch_size, 1, hidden_size)
        
        # Concatenating output from LSTM with context vector
        output = torch.cat((rnn_out, context_vector.permute(1,0,2)),-1)
        # output shape: (1, batch_size, 2*hidden_size)
        
        predictions = self.Linear(output)
        # predictions shape: (1, batch_size, length_target_vocabulary)
        
        # predictions shape: (1, batch_size, length_target_vocabulary) to send it to
        # loss function we want it to be (batch_size, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)
        return predictions, hidden, attn_weights
    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden shape: (1, batch_size, hidden_size) 
        
        encoder_outputs shape: (sequence_length, batch_size, hidden_size)
        """
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            # (batch_size, sequence_lenght, 1)
            return torch.bmm(encoder_outputs.permute(1,0,2), decoder_hidden.permute(1,2,0))

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden.squeeze())
            # out shape:(batch_size, hidden_size)
    
            # (batch_size, sequence_lenght, 1)
            return torch.bmm(encoder_outputs.permute(1,0,2), out.unsqueeze(2))

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden.permute(1,0,2)+encoder_outputs.permute(1,0,2)))
            # out shape:(batch_size, sequence_length, hidden_size)
            
            # (batch_size, sequence_lenght, 1)
            return torch.bmm(out, self.weight.repeat(out.shape[0],1,1).permute(0,2,1))
