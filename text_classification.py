import os
import matplotlib.pyplot as plt
from sklearn import *
import sklearn


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
    return preprocessed_dataset, dataset.target


# T1Q5
# splitting the set into train_set & test_set
def split_test_set():
    preprocessed_data, class_indices = preprocess_data()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(preprocessed_data, class_indices, train_size=0.8, test_size=0.2, random_state=None, shuffle=False, stratify=None)
    return X_train, X_test, y_train, y_test


# T1Q6
# train classifier with training set and use it on test set
def nb_classifier(alpha=1.0):
    counter = 0
    X_train, X_test, y_train, y_test = split_test_set()
    classifier = sklearn.naive_bayes.MultinomialNB(alpha=alpha)
    classifier.fit(X_train, y_train)
    for index in range(50):
        predicted_index = classifier.predict(X_test[index])[0]
        print("Predicted")
        print(class_names[predicted_index])
        print("Actual")
        print(class_names[y_test[index]])
        if predicted_index == y_test[index]:
            counter += 1
    print(counter)


def write_results_to_file():
    with open("./Output/bbc-distribution.txt", "a") as f:
        f.write("\(a\) ********MultinomialNB default values, try 1********\n")
        f.close()