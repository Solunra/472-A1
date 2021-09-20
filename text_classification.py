import os
import matplotlib.pyplot as plt
from sklearn import *
import sklearn


dataset_folder = "./Documents/BBC/"


def preprocess_data():
    dataset = sklearn.datasets.load_files(dataset_folder, encoding="latin1")
    preprocessed_dataset = sklearn.feature_extraction.text.CountVectorizer(dataset, encoding="latin1")
    return preprocessed_dataset


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


# Generates the PDF plot of the sub folder file count
def generate_pdf_distribution_of_instance_distribution():
    value_dict = explore_sub_folders_count_files()
    plt.plot(value_dict.keys(), value_dict.values())
    plt.savefig("./Output/BBC-distribution.pdf")
