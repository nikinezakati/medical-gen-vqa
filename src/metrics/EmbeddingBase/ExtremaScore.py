import torch
from src.metrics.EmbeddingBase.EmbeddingBaseMetric import EmbeddingBaseMetric


class ExtremaScore(EmbeddingBaseMetric):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def __get_extrema(self, sentence):
        """
        Compute the Extrema vector from a tensor of embedding vectors.
        :param sentence: tensor of shape (seq_len, embedding_size)
        :return: the Extrema vector.
        """
        
        max_values = torch.max(sentence, dim=0).values
        min_values = torch.min(sentence, dim=0).values
        return torch.tensor([
            min_values[min_v_i] if torch.abs(min_values[min_v_i]) > max_values[max_v_i] else max_values[max_v_i]
            for min_v_i, max_v_i in zip(range(min_values.shape[0]), range(max_values.shape[0]))
        ])


    @torch.no_grad()
    def extrema_sentence_level(self, hypothesis_sentence, reference_sentence):
        """
        Compute Extrema on sentence level.
        :param hypothesis_sentence:
        :param reference_sentence:
        :return:
        """
        hypothesis = hypothesis_sentence
        reference = reference_sentence
        return self._cos_sim(
            a=self.__get_extrema(hypothesis),
            b=self.__get_extrema(reference),
        )


    @torch.no_grad()
    def compute(self, hypothesis_corpus, reference_corpus):
        """
        Compute Extrema on corpus level.
        :param hypothesis_corpus:
        :param reference_corpus:
        :return:
        """
        scores = []
        for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):
            X = hypothesis
            Y = reference

            value = self._cos_sim(self.__get_extrema(X), self.__get_extrema(Y))
            scores.append(value)
        scores = torch.stack(scores, dim = 0)
        return self._compute_corpus_score(scores)