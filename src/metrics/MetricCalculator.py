import torch
import evaluate
from src.metrics.EmbeddingBase.AverageScore import AverageScore
from src.metrics.EmbeddingBase.ExtremaScore import ExtremaScore
from src.metrics.EmbeddingBase.GreedyMatchingScore import GreedyMatchingScore
from src.metrics.cider.cider import Cider
#!pip install evaluate
#!pip install rouge_score
#!pip install bert_score

class MetricCalculator():
    def __init__(self, embedding_layer) -> None:

        self.embedding_layer = embedding_layer
        self.METRICS = ["average_score", "bleu", "rougeL", "meteor", "bertscore"]
        self.accumelated_instances = []
        
        #overlapping ngram metircs
        self.BLEU = evaluate.load('bleu')
        self.ROUGE = evaluate.load('rouge')
        self.METEOR = evaluate.load('meteor')
        self.BERTSCORE = evaluate.load("bertscore")
        self.CIDEr = Cider()

    def add_batch(self, preds, references, preds_ids, ref_ids):
        
        # compute embedding based metrics
        metrics = {
            "average_score" : AverageScore(),
            # "extrema_score" : ExtremaScore(),
            # "greedy_matching_score" : GreedyMatchingScore()
        }

        result = {}
        # tokenize inputs, we have to ignore [SEP] , [START] tokens. 
        # preds_tokenized = [self.tokenizer(pred, return_tensors="pt")['input_ids'].squeeze()[1:-1].cuda() for pred in preds]
        # ref_tokenized = [self.tokenizer(ref, return_tensors="pt")['input_ids'].squeeze()[1:-1].cuda() for ref in references]
        
        preds_emb = [self.embedding_layer(torch.tensor(s, dtype=torch.int).cuda()) for s in preds_ids]
        ref_emb = [self.embedding_layer(torch.tensor(s, dtype=torch.int).cuda()) for s in ref_ids]
        
        for key in metrics:
            result[key] = metrics[key].compute(preds_emb, ref_emb)
        
        self.BLEU.add_batch(predictions=preds, references=references)
        self.ROUGE.add_batch(predictions=preds, references=references)
        self.METEOR.add_batch(predictions=preds, references=references)
        self.BERTSCORE.add_batch(predictions=preds, references=references)

        result[self.CIDEr.method()] = self.CIDEr.compute_score(preds_ids, ref_ids)

        self.accumelated_instances.append(result)
        return result

    def compute(self):
        result = {}
        result[self.BLEU.name] = self.BLEU.compute()
        result[self.ROUGE.name] = self.ROUGE.compute()
        result[self.METEOR.name] = self.METEOR.compute()
        result[self.BERTSCORE.name] = self.BERTSCORE.compute(model_type="microsoft/deberta-xlarge-mnli")
        
        avg_bert_keys = ['precision', 'recall', 'f1']
        
        for key in avg_bert_keys:
            result[self.BERTSCORE.name][key] = sum(result[self.BERTSCORE.name][key]) / len(result[self.BERTSCORE.name][key])
        
        average_scores = []
        extrema_scores = []
        greedy_scores = []
        cider_scores = []
        
        
        #compute other metrics manually
        for item in self.accumelated_instances:
            average_scores.append(item['average_score'].mean)
            # extrema_scores.append(item['extrema_score'].mean)
            # greedy_scores.append(item['greedy_matching_score'].mean)
            cider_scores.append(item[self.CIDEr.method()][0])
        
        result['average_score'] = torch.mean(torch.stack(average_scores)).cpu().tolist()
        # result['extrema_score'] = torch.mean(torch.stack(extrema_scores))
        # result['greedy_matching_score'] = torch.mean(torch.stack(greedy_scores))
        result[self.CIDEr.method()] = sum(cider_scores) / len(cider_scores)
        return result



if __name__ == '__main__':
    from transformers import LxmertTokenizer, LxmertModel

    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
    E = model.embeddings.word_embeddings.cuda()

    mc = MetricCalculator(E)
    preds = [['the cat is on the mat', 'the cat is on the mat'],  ['the cat is on the mat', 'the cat is on the mat']]
    target = [['there is a catty on the matew', 'a cat is on the mat'], ['there is a catty on the matew', 'a cat is on the mat']]
    for pred, ref in zip(preds, target):
        preds_tokenized = [tokenizer(predd)['input_ids'][1:-1] for predd in pred]
        ref_tokenized = [tokenizer(reff)['input_ids'][1:-1] for reff in ref]
        mc.add_batch(pred, ref, preds_tokenized, ref_tokenized)
    print(mc.compute())