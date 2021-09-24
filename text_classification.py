import os
import matplotlib.pyplot as plt
from sklearn import *
import sklearn
import math

dataset_folder = "./Documents/BBC/"
class_names = []


# T1Q2
# explores the sub folders under Documents/BBC
# returns a dictionary of class => numOfInstances
def explore_sub_folders_count_files():
    value_dict = {}
    for root, dirs, _ in os.walk(dataset_folder):
        # sub_dir will be the class
        for sub_dir in dirs:
            # each file is an instance of the sub_dir class
            for _, _, file in os.walk(root + sub_dir):
                value_dict[sub_dir] = len(file)

    return value_dict


# T1Q2
# Generates the PDF plot of the sub folder file count
def generate_pdf_distribution_of_instance_distribution():
    value_dict = explore_sub_folders_count_files()
    plt.plot(value_dict.keys(), value_dict.values())
    plt.savefig("./Output/BBC-distribution.pdf")


# T1Q3, T1Q4
# loading corpus, preprocessing data
def preprocess_data():
    global class_names
    dataset = sklearn.datasets.load_files(dataset_folder, encoding="latin1")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding="latin1")
    preprocessed_dataset = vectorizer.fit_transform(dataset['data'])
    class_names = dataset['target_names']
    # Returns document-term matrix
    # (documents on the row header, words on the column header. The rest is frequency of each word for each document)
    # and target labels' index where ['business', 'entertainment', 'politics', 'sport', 'tech'] = [0, 1, 2, 3, 4]
    return preprocessed_dataset, dataset.target


# T1Q5
# splitting the set into train_set & test_set
def split_test_set():
    preprocessed_data, class_indices = preprocess_data()
    # the x list's values are the inputs with the y list's values being the output
    # in this case, x are the document-term matrix and y are the class' index (see preprocess_data())
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(preprocessed_data, class_indices, train_size=0.8, test_size=0.2, random_state=None, shuffle=False, stratify=None)
    return preprocessed_data, X_train, X_test, y_train, y_test


# T1Q6
# train classifier with training set and use it on test set
def nb_classifier(alpha=1.0):
    preprocessed_data, X_train, X_test, y_train, y_test = split_test_set()
    classifier = sklearn.naive_bayes.MultinomialNB(alpha=alpha)
    classifier.fit(X_train, y_train)
    return classifier, preprocessed_data, X_train, X_test, y_train, y_test


#T1Q7
def write_results_to_file():
    with open("./Output/bbc-distribution.txt", "w") as f:
        f.write('(a) ********MultinomialNB default values, try 1********\n')
        classifier, preprocessed_data, X_train, X_test, y_train, y_test = nb_classifier()
        predicted_y = classifier.predict(X_test)
        # row is predicted value, column is actual value
        # matrix values are what was predicted and what it is in count
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predicted_y)
        f.write(f'(b)\n{confusion_matrix}\n')
        classification_report = sklearn.metrics.classification_report(y_test, predicted_y, output_dict=True)
        f.write(f'(c)\n{classification_report}\n')
        accuracy_score = sklearn.metrics.accuracy_score(y_test, predicted_y)
        # failed to add labels, it uses the index instead
        f1_score_macro = sklearn.metrics.f1_score(y_test, predicted_y, average='macro')
        f1_score_weighted = sklearn.metrics.f1_score(y_test, predicted_y, average='weighted')
        f.write(f'(d) accuracy score: {accuracy_score}\n'
                f'    macro f1 score: {f1_score_macro}\n'
                f'    weighted f1 score: {f1_score_weighted}\n')
        f.write(f'(e)\nlogarithmic prior probability of classes 0 to {len(classifier.class_log_prior_)}: {classifier.class_log_prior_}\n')
        f.write(f'prior probability of:\n')

        for index, class_ in enumerate(classifier.classes_):
            f.write(f'    {class_}: {math.exp(classifier.class_log_prior_[index])}\n')
        f.write(f'(f) size of the vocabulary: {preprocessed_data.shape[1]}\n')
        # find the number of word-token, zero entries and non-zero for each class
        classes_num_words = [0] * len(classifier.classes_)
        classes_num_non_zero = [0] * len(classifier.classes_)
        classes_num_zero = [0] * len(classifier.classes_)
        f.write(f'(g) for every class:\n')

        for index, y_train_instance in enumerate(y_train):
            classes_num_words[y_train_instance] += X_train[index].sum()
            classes_num_non_zero[y_train_instance] += X_train[index].getnnz()
            classes_num_zero[y_train_instance] += (X_train[index].getnnz() - X_train[index].count_nonzero())
        
        for index, class_num_word in enumerate(classes_num_words):
            f.write(f'class {class_names[index]} has {class_num_word} word-tokens\n')

        f.write(f'(h) number of word-tokens in the entire corpus: {preprocessed_data.sum()}\n')
        
        f.write(f'(i) for every class:\n')
        for index, _ in enumerate(classes_num_words):
            f.write(f'class {class_names[index]} has {classes_num_zero[index]} words that do not occur. It has a frequency of \'zero\' words of {classes_num_zero[index]/(classes_num_zero[index]+classes_num_non_zero[index])}\n')
