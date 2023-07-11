
import argparse
import numpy as np
import random

def sigmoid(x):
  
    s = 1 / (1 + np.exp(-x))

    return s


def normalizeRows(x):
  
    N = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((N,1)) + 1e-30
    return x

def softmax(x):

    orig_shape = x.shape

    if len(x.shape) > 1:
    
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
   
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp

    assert x.shape == orig_shape
    return x

def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
   
    softmax_vector = softmax(np.dot(outsideVectors, centerWordVec)) 
    loss = -np.log(softmax_vector[outsideWordIdx])  
    gradCenterVec = -outsideVectors[outsideWordIdx] + np.dot(np.transpose(outsideVectors), softmax_vector)  #shape (D, )
    gradOutsideVecs = np.dot(softmax_vector.reshape((softmax_vector.shape[0], 1)), np.transpose(centerWordVec.reshape((centerWordVec.shape[0], 1))))   #shape (N, D)
    gradOutsideVecs[outsideWordIdx] -= centerWordVec


    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    sigmoid_value = sigmoid(-np.dot(outsideVectors[negSampleWordIndices, :], centerWordVec.reshape((centerWordVec.shape[0], 1)))) #shaoe (K, 1)
    outsider = -np.log(sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec)))
    samples = -np.sum(np.log(sigmoid_value), axis=0)
    loss = outsider + samples
    outsider = -(1 - sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))) * outsideVectors[outsideWordIdx]
    samples = np.sum((1 - sigmoid_value)*outsideVectors[negSampleWordIndices, :], axis=0)
    gradCenterVec = outsider + samples
    gradOutsideVecs = np.zeros(outsideVectors.shape)
    outsiders_matrix = (1 - sigmoid_value) * centerWordVec

    for idx, value in enumerate(negSampleWordIndices):
        gradOutsideVecs[value] += outsiders_matrix[idx]
    gradOutsideVecs[outsideWordIdx] = -(1 - sigmoid(np.dot(outsideVectors[outsideWordIdx], centerWordVec))) * centerWordVec

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
 

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)


    for outsideWord in outsideWords:
        centerWordIdx = word2Ind[currentCenterWord]
        centerWordVec = centerWordVectors[centerWordIdx]
        outsideWordIdx = word2Ind[outsideWord]
        loss_t, gradCenterVec_t, gradOutsideVectors_t = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += loss_t
        gradCenterVecs[centerWordIdx] = np.add(gradCenterVecs[centerWordIdx], gradCenterVec_t)
        gradOutsideVectors = np.add(gradOutsideVectors, gradOutsideVectors_t)
 
    
    return loss, gradCenterVecs, gradOutsideVectors


def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad