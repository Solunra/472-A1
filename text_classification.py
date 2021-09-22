import os
import matplotlib.pyplot as plt
from sklearn import *
import sklearn


dataset_folder = "./Documents/BBC/"


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
    dataset = sklearn.datasets.load_files(dataset_folder, encoding="latin1")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(encoding="latin1")
    preprocessed_dataset = vectorizer.fit_transform(dataset['data'])
    return preprocessed_dataset, dataset.target


# T1Q5
# splitting the set into train_set & test_set
def split_test_set():
    preprocessed_data, class_indices = preprocess_data()
    train_set, test_set = sklearn.model_selection.train_test_split(preprocessed_data, train_size=0.8, test_size=0.2, random_state=None, shuffle=False, stratify=None)
    return train_set, test_set, class_indices


# T1Q6
# train classifier with training set and use it on test set
def nb_classifier():
    train_set, test_set, class_indices = split_test_set()
    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(train_set,class_indices)

nb_classifier()