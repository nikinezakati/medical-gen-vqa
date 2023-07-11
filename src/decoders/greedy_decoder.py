import torch

class GreedyDecoder():
    def __init__(self, tokenizer, pad = 0, sep = 102):
        self.tokenizer = tokenizer
        self.SEP = sep
        self.PAD = pad
    
    def decode_from_logits(self, logits):
        logits = torch.argmax(logits, dim=-1)    
        return logits

    def batch_decode(self, tokens):
        
        # return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

        sentences = []
        sentences_ids = []
        for i in range(tokens.shape[0]):
            sentence = []
            for j in range(tokens.shape[1]):
                if(tokens[i, j] == self.PAD):
                    continue
                if(tokens[i, j] == self.SEP):
                    break
                sentence.append(tokens[i, j])
            sentences.append(self.tokenizer.decode(sentence, skip_special_tokens=True))
            sentences_ids.append(sentence)
        return sentences, sentences_ids