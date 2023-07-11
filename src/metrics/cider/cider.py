# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
# Edited: Hadi Sheikhi (ha_sheikhi@comp.iust.ac.ir) 

import pdb
from .cider_scorer import CiderScorer

class Cider:
    """
    Main Class to compute the CIDEr metric

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_sentences : list of <tokenized hypothesis / candidate sentence>
                ref_sentences  : list of <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert(len(gts) == len(res))

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for hypo, ref in zip(gts, res):


            # Sanity check.
            assert(type(hypo) is list)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo, ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"
