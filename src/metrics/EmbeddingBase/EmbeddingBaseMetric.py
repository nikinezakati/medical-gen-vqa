import collections
import torch

CorpusLevelScore = collections.namedtuple('CorpusLevelScore',
                                          ['mean', 'confidence_interval', 'standard_deviation'])

class EmbeddingBaseMetric():
    def __init__(self):

        # See https://en.wikipedia.org/wiki/1.96 for details of this magic number.
        self.__95_CI_DEVIATE = 1.96

        self._EPSILON = 0.00000000001
    
    @torch.no_grad()
    def _compute_corpus_score(self, scores):
        """
        Compute various statistics from a list of scores.
        The scores come from evaluating a list of sentence pairs.
        The function combines them by mean and standard derivation.
        :param scores: tensor of floats.
        :return: a CorpusLevelScore.
        """
        return CorpusLevelScore(
            mean=torch.mean(scores),
            confidence_interval=self.__95_CI_DEVIATE * torch.std(scores) / scores.shape[0],
            standard_deviation=torch.std(scores),
        )
    
    @torch.no_grad()
    def _cos_sim(self, a, b):
        """
        Return the cosine similarity of two tensors a and b.
        :param a: tensor of 1D of size (embedding_size).
        :param b: tensor of 1D of size (embedding_size).
        :return: float tensor.
        """
        a_norm = torch.norm(a)
        b_norm = torch.norm(b)
        if a_norm < self._EPSILON or b_norm < self._EPSILON:
            # zero in, zero out.
            return 0
        return torch.dot(a, b) / a_norm / b_norm