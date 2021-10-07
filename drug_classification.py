import pandas
import matplotlib.pyplot as plt
from sklearn import *
import sklearn


dataset_file = "./Documents/drug200.csv"


# T2Q2; Class will be 'Drug' types
def generate_pdf_of_instance_distribution(csv_data):
    csv_data = pandas.read_csv(dataset_file)
    class_names = csv_data['Drug'].unique().tolist()
    class_names.sort()
    instance_number = csv_data.groupby('Drug').size().tolist()
    plt.plot(class_names, instance_number)
    plt.savefig("./Output/drug-distribution.pdf")


# T2Q4; Need to convert Sex:[M, F], BP:[L, N, H], Cholesterol:[L, N, H], and Drug:[drugA, drugB, drugC, drugX, drugY] to numerical format
def convert_to_numerical_format():
    csv_data = pandas.read_csv(dataset_file)
    # unordered values
    set_of_sex_values = pandas.Categorical(csv_data['Sex'], ordered=False)
    sex_columns = pandas.get_dummies(set_of_sex_values, prefix='Sex')
    csv_data = csv_data.drop(labels='Sex', axis=1)
    csv_data = csv_data.join(sex_columns)
    # ordered values; BP shares values with Cholesterol
    # reason for sorting is to make it consistent with PDF's order
    set_of_bp_values = pandas.Categorical(csv_data['BP'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH'])
    ordered_bp_list = set_of_bp_values.categories.tolist()

    processed_data_frame = csv_data.replace(to_replace=ordered_bp_list, value=range(len(ordered_bp_list)))
    # dropped to put into separate list
    processed_data_frame = processed_data_frame.drop(labels='Drug', axis=1)
    return processed_data_frame, csv_data['Drug'].tolist()


def prep_classifier_for_analysis():
    processed_dataset, classes_index_dataset = convert_to_numerical_format()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(processed_dataset, classes_index_dataset)
    return x_train, x_test, y_train, y_test


def run_classifiers():
    x_train, x_test, y_train, y_test = prep_classifier_for_analysis()
    nb_classifier = sklearn.naive_bayes.MultinomialNB()
    nb_classifier.fit(x_train, y_train)
    nb_predicted_y = nb_classifier.predict(x_test)
    nb_confusion_matrix = sklearn.metrics.confusion_matrix(y_test, nb_predicted_y)
    nb_classification_report = sklearn.metrics.classification_report(y_test, nb_predicted_y, output_dict=True)
    nb_accuracy_score = sklearn.metrics.accuracy_score(y_test, nb_predicted_y)
    nb_f1_score_macro = sklearn.metrics.f1_score(y_test, nb_predicted_y, average='macro')
    nb_f1_score_weighted = sklearn.metrics.f1_score(y_test, nb_predicted_y, average='weighted')

    # b_dt_classifier = sklearn.tree.DecisionTreeClassifier()
    # b_dt_classifier.fit(x_train, y_train)
    # b_dt_predicted_y = b_dt_classifier.fit(x_train, y_train)
    # b_dt_confusion_matrix = sklearn.metrics.accuracy_score(y_test, b_dt_predicted_y)
    # b_dt_classification_report = sklearn.metrics.classification_report(y_test, b_dt_predicted_y, output_dict=True)
    # b_dt_accuracy_score = sklearn.metrics.accuracy_score(y_test, b_dt_predicted_y)
    # b_dt_f1_score_macro = sklearn.metrics.f1_score(y_test, b_dt_predicted_y, average='macro')
    # b_dt_f1_score_weighted = sklearn.metrics.f1_score(y_test, b_dt_predicted_y, average='weighted')

run_classifiers()