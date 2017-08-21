import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    N: number of data points
    p: number of parameters
    Assume 
        # of features = d
        # of HMM states = n
    Then
    p = 
        # of probabilities in transition matrix + 
        # of probabilities in initial distribution + 
        # of Gaussian mean + 
        # of Gaussian variance 
      = 
        n*(n-1) + (n-1) + 2*d*n
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        all_n_components = []
        all_scores = [] # Store each BIC value
        N, d = self.X.shape
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                bic = -2 * logL + (n_components*(n_components-1) + (n_components-1) + 2*d*n_components) * np.log(N)
                all_scores.append(bic)
                all_n_components.append(n_components)
            except:
                # eliminate non-viable models from consideration
                pass

        best_num_components = all_n_components[np.argmin(all_scores)] if all_scores else self.n_constant
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    DIC Equation:
        DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
            (Equation (17) in Reference [0]. Assumes all data sets are same size)
            = log likelihood of the data belonging to model
              - avg of anti log likelihood of data X and model M
            = log(P(original word)) - average(log(P(other words)))
        
            where anti log likelihood means likelihood of data X and model M belonging to competing categories
            where log(P(X(i))) is the log-likelihood of the fitted model for the current word
            (in terms of hmmlearn it is the model's score for the current word)
            where where "L" is likelihood of data fitting the model ("fitted" model)
            where X is input training data given in the form of a word dictionary
            where X(i) is the current word being evaluated
            where M is a specific model

            Note:
                - log likelihood of the data belonging to model
                - anti_log_likelihood of data X vs model M
    
    Selection using DIC Model:
        - Higher the DIC score the "better" the model.
        - SelectorDIC accepts argument of ModelSelector instance of base class
          with attributes such as: this_word, min_n_components, max_n_components,
        - Loop from min_n_components to max_n_components
        - Find the highest BIC score as the "better" model.
    '''

    # Calculate anti log likelihoods.
    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]

    def calc_best_score_dic(self, score_dics):
        # Max of list of lists comparing each item by value at index 0
        return max(score_dics, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = []
        models = []
        score_dics = []
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(num_states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))

        # Note: Situation that may cause exception may be if have more parameters to fit
        # than there are samples, so must catch exception when the model is invalid
        except Exception as e:
            # logging.exception('DIC Exception occurred: ', e)
            pass
        for index, model in enumerate(models):
            log_likelihood_original_word, hmm_model = model
            score_dic = log_likelihood_original_word - np.mean(self.calc_log_likelihood_other_words(model, other_words))
            score_dics.append(tuple([score_dic, model[1]]))
        return self.calc_best_score_dic(score_dics)[1] if score_dics else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        all_n_components = []
        split_method = KFold()
        all_scores = [] # Store each CV value
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                if len(self.sequences) > 2: # Check if there are enough data to split
                    scores = []
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Prepare training sequences
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        # Prepare testing sequences
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        model = self.base_model(n_components)
                        scores.append(model.score(X_test, lengths_test))
                    all_scores.append(np.mean(scores))
                else:
                    model = self.base_model(n_components)
                    all_scores.append(model.score(self.X, self.lengths))
                all_n_components.append(n_components)
            except:
                # eliminate non-viable models from consideration
                pass

        best_num_components = all_n_components[np.argmax(all_scores)] if all_scores else self.n_constant
        return self.base_model(best_num_components)
