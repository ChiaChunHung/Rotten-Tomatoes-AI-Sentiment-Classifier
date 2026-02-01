#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_d = defaultdict(int)  # create a dict with initial value set to 0
    for ch in x.split():

        word_d[ch] += 1        # defaultdict() 讓你可以造出可以同時做新增key和抓到已經存在的key並增加value的動作
    return word_d
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!

    In Milestone 5, we will also require you to call evaluatePredictor()
    on both trainExamples and validationExamples to see how you're doing
    as you learn after each epoch. Note also that the identity function
    may be used as the featureExtractor function during testing.
    """
    weights = {}  # the weight vector

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for epoch in range(numEpochs):
        # loop over every data to update the weight
        for x, y in trainExamples:
            y = 0 if y < 0 else 1
            phi_x = featureExtractor(x)
            k = dotProduct(weights, phi_x)
            h = 1/(1+math.exp(-k))  # h = sigmoid(k) = 1/(1+math.exp(-k))
            scale = -alpha*(h-y)
            increment(weights, scale, phi_x)

        # create predictor
        def predictor(review):
            """
            @param review: str, it's either x from trainExamples or validationExamples
            @return: int, prediction y' of review. It's either +1 or -1
            """
            phi_vector = featureExtractor(review)
            score = dotProduct(phi_vector, weights)
            return 1 if score >= 0 else -1
        # show Training Error and Validation Error
        print(f'Training Error: ({epoch} epoch): {evaluatePredictor(trainExamples, predictor)}')
        print(f'Validation Error: ({epoch} epoch): {evaluatePredictor(validationExamples, predictor)}')

    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        phi = defaultdict(int)
        for i in range(random.randint(1, len(weights))):  # 從0有可能會連一次迴圈都不執行，這會導致沒有例子產出
            phi[list(weights.keys())[random.randint(0, len(weights)-1)]] += 1
        y = 1 if dotProduct(phi, weights) >= 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        word_d = defaultdict(int)
        x_without_space = ''.join(x.split())
        for i in range(len(x_without_space)-n+1):
            word_d[x_without_space[i:i+n]] += 1
        return word_d
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

