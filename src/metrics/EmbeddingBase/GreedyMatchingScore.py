import torch
from src.metrics.EmbeddingBase.EmbeddingBaseMetric import EmbeddingBaseMetric

class GreedyMatchingScore(EmbeddingBaseMetric):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def __greedy_match(self, a, b):
        """
        Perform the greedy match on two list of word vectors.
        See photos/greedy matching.png.
        :param a: tensor of shape (seq_len, embedding_size)
        :param b: tensor of shape (seq_len, embedding_size)
        :return: The greedy-matched value.
        """
        sum_max_cosine = sum(
            max(
                self._cos_sim(a[a_i], b[b_i]) for b_i in range(b.shape[0])
            ) for a_i in range(a.shape[0])
        )

        return sum_max_cosine / a.shape[0]


    @torch.no_grad()
    def __greedy_average(self, a, b):
        """
        Compute the average of greedy matching a on b and b on a.
        :param a: tensor of shape (seq_len, embedding_size)
        :param b: tensor of shape (seq_len, embedding_size)
        :return: The averaged greedy-matched value.
        """
        # return np.mean([_greedy_match(*args) for args in ((a, b), (b, a))])
        return (self.__greedy_match(a, b) + self.__greedy_match(b, a)) / 2


    @torch.no_grad()
    def greedy_match_sentence_level(self, hypothesis_sentence, reference_sentence):
        """
        Compute Greedy Matching on sentence level.
        :param hypothesis_sentence:
        :param reference_sentence:
        :return:
        """
        hyp = hypothesis_sentence
        ref = reference_sentence
        return self.__greedy_average(hyp, ref)


    @torch.no_grad()
    def compute(self, hypothesis_corpus, reference_corpus):
        """
        Compute Greedy Matching on corpus level.
        :param hypothesis_corpus:
        :param reference_corpus:
        :return:
        """
        scores = []
        for hypothesis, reference in zip(hypothesis_corpus, reference_corpus):
            X = hypothesis
            Y = reference
            if X.shape[0] == 0 or Y.shape[0] == 0:
                scores.append(0)
                continue
            scores.append(self.__greedy_average(X, Y))
        scores = torch.stack(scores, dim = 0)
        return self._compute_corpus_score(scores)