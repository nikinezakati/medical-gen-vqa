import torch
from src.metrics.EmbeddingBase.EmbeddingBaseMetric import EmbeddingBaseMetric

class AverageScore(EmbeddingBaseMetric):
    def __init__(self):
        super().__init__()       
    
    @torch.no_grad()
    def __embedding_sum(self, sentence):
        """
        Return the sum of embeddings of words in sentences.
        :param sentence:  (seq_len, embedding_size)
        :return: a 1D tensor of size (embedding_size).
        """
        total = torch.sum(sentence, dim=0)
        return total

    @torch.no_grad()
    def __get_average(self, sentence):
        total =  self.__embedding_sum(sentence)
        total_norm = torch.norm(total)
        return total / total_norm


    @torch.no_grad()
    def average_sentence_level(self, hypothesis_sentence, reference_sentence):
        """
        Compute Average on sentence level.
        :param hypothesis_sentences:
        :param reference_sentences:
        :return:
        """
        return self._cos_sim(
            a=self.__get_average(hypothesis_sentence),
            b=self.__get_average(reference_sentence),
        )

    @torch.no_grad()
    def compute(self, hypothesis_corpus, reference_corpus):
        """
        Compute Average on corpus level.
        :param hypothesis_corpus:
        :param reference_corpus:
        :return:
        """
        assert len(hypothesis_corpus) == len(reference_corpus)
        scores = []

        for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):
            X = self.__embedding_sum(hypothesis)
            Y = self.__embedding_sum(reference)
            # if none of the words in ground truth have embeddings, skip
            if torch.norm(X) < self._EPSILON:
                continue

            # if none of the words have embeddings in response, count result as zero
            if torch.norm(Y) < self._EPSILON:
                scores.append(0)
                continue

            # Normalize to unit vectors.
            X /= torch.norm(X)
            Y /= torch.norm(Y)
            scores.append(self._cos_sim(X, Y))
        scores = torch.stack(scores, dim=0)
        return self._compute_corpus_score(scores)
