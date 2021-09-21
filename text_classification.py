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
    print(vectorizer.get_feature_names())
    return preprocessed_dataset


# T1Q5
# splitting the set into train_set & test_sest
def split_test_set():
    all_data = preprocess_data()
    train_set, test_set = sklearn.model_selection.train_test_split(all_data, test_size=0.2, train_size = 0.8, random_state = None, shuffle = False, stratify = None)
    print("The current train set is:")
    print(train_set)
    print("The current test set is:")
    print(test_set)
    return train_set, test_set
