import collections

import numpy as np

from .svm import *
from .util import *


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return [x.lower() for x in message.split(" ")]
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    counter = collections.Counter()
    for msg in messages:
        words = get_words(msg)
        for word in set(words):
            counter[word] += 1

    _map = {}
    for word, count in counter.items():
        if count >= 5:
            _map[word] = len(_map)

    return _map
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    mat = np.zeros((len(messages), len(word_dictionary)))
    for i, sentence in enumerate(messages):
        words = get_words(sentence)
        for word in words:
            _id = word_dictionary.get(word, None)
            if _id is not None:
                mat[i, _id] += 1
    return mat
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    _, V = matrix.shape

    model = {}
    model["phi_x_y"] = np.zeros((V, 2))

    for label in [0, 1]:
        model["phi_x_y"][:, label] = 1 + matrix[labels == label].sum(axis=0)
        model["phi_x_y"][:, label] = model["phi_x_y"][:, label] / (
            model["phi_x_y"][:, label].sum() + V
        )

    model["phi_y"] = np.asarray([1 - labels.mean(), labels.mean()])

    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    pred = matrix @ np.log(model["phi_x_y"])
    pred = pred + np.log(model["phi_y"])
    pred = np.exp(pred)
    pred = pred[:, 1] / pred.sum(axis=1)
    return pred > 0.5
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    top_5_ids = np.argsort(model["phi_x_y"][:, 1] / model["phi_x_y"][:, 0])[-5:]
    rev_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return [rev_dictionary[_id] for _id in top_5_ids]
    # *** END CODE HERE ***


def compute_best_svm_radius(
    train_matrix, train_labels, val_matrix, val_labels, radius_to_consider
):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accs = []
    for r in radius_to_consider:
        output = train_and_predict_svm(train_matrix, train_labels, val_matrix, r)
        accs.append((output == val_labels).mean())

    return radius_to_consider[np.argmax(accs)]

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = load_spam_dataset("spam_train.tsv")
    val_messages, val_labels = load_spam_dataset("spam_val.tsv")
    test_messages, test_labels = load_spam_dataset("spam_test.tsv")

    dictionary = create_dictionary(train_messages)

    print("Size of dictionary: ", len(dictionary))

    write_json("spam_dictionary", dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt("spam_sample_train_matrix", train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(
        naive_bayes_model, test_matrix
    )

    np.savetxt("spam_naive_bayes_predictions", naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print(
        "Naive Bayes had an accuracy of {} on the testing set".format(
            naive_bayes_accuracy
        )
    )

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print("The top 5 indicative words for Naive Bayes are: ", top_5_words)

    write_json("spam_top_indicative_words", top_5_words)

    optimal_radius = compute_best_svm_radius(
        train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10]
    )

    write_json("spam_optimal_radius", optimal_radius)

    print("The optimal SVM radius was {}".format(optimal_radius))

    svm_predictions = train_and_predict_svm(
        train_matrix, train_labels, test_matrix, optimal_radius
    )

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print("The SVM model had an accuracy of {} on the testing set".format(svm_accuracy))


if __name__ == "__main__":
    main()
