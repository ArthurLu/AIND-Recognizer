import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for sequence, length in test_set.get_all_Xlengths().values():
        probability = {}
        best_guess, best_score = None, float("-inf")
        for word, model in models.items():
            try:
                logL = model.score(sequence, length)
                probability[word] = logL
                if logL > best_score:
                    best_guess, best_score = word, logL
            except:
                # eliminate non-viable models from consideration
                probability[word] = float("-inf")
                pass
        guesses.append(best_guess)
        probabilities.append(probability)
    return probabilities, guesses
