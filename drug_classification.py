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


def preprocess_data():
    # T2Q4; Need to convert Sex:[M, F], BP:[L, N, H] and Cholesterol:[L, N, H] to numerical format
    def convert_to_numerical_format():  # inner function since it's only used by prep_classifier_for_analysis
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

    processed_dataset, classes_index_dataset = convert_to_numerical_format()
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(processed_dataset, classes_index_dataset)
    return x_train, x_test, y_train, y_test


def run_classifiers():
    def get_performance_metrics(classifier, x_test, y_test):
        metrics = {}
        predicted_y = classifier.predict(x_test)
        metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_test, predicted_y)
        metrics['classification_report'] = sklearn.metrics.classification_report(y_test, predicted_y, output_dict=True)
        metrics['accuracy_score'] = sklearn.metrics.accuracy_score(y_test, predicted_y)
        metrics['f1_score_macro'] = sklearn.metrics.f1_score(y_test, predicted_y, average='macro')
        metrics['f1_score_weighted'] = sklearn.metrics.f1_score(y_test, predicted_y, average='weighted')
        return metrics

    x_train, x_test, y_train, y_test = preprocess_data()
    
    # 6.a Gaussian Naive Bayes
    nb_classifier = sklearn.naive_bayes.MultinomialNB()
    nb_classifier.fit(x_train, y_train)
    nb_metrics = get_performance_metrics(nb_classifier, x_test, y_test)

    # 6.b Base-DT
    b_dt_classifier = sklearn.tree.DecisionTreeClassifier()
    b_dt_classifier.fit(x_train, y_train)
    b_dt_metrics = get_performance_metrics(b_dt_classifier, x_test, y_test)

    # 6.c Top-DT
    # values required to play around with:
    # tree_parameters = {'criterion': ['gini' OR 'entropy'], 'max_depth': [2 values], 'min_samples_split': [3 values]}
    tree_parameters = {'criterion': ['entropy'], 'max_depth': [5, 10], 'min_samples_split': [2, 4, 6]}
    t_raw_dt_classifier = sklearn.tree.DecisionTreeClassifier()
    t_dt_classifier = sklearn.model_selection.GridSearchCV(t_raw_dt_classifier, tree_parameters)
    t_dt_classifier.fit(x_train, y_train)
    t_dt_metrics = get_performance_metrics(t_dt_classifier, x_test, y_test)

    # 6.d Perceptron
    perceptron_classifier = sklearn.linear_model.Perceptron()
    perceptron_classifier.fit(x_train, y_train)
    perceptron_metrics = get_performance_metrics(perceptron_classifier, x_test, y_test)

    # 6.e Base Multi-Layered Perceptron
    # hidden_layer_sizes=[100] -> a single hidden layer with 100 neurons
    # logistic -> sigmoid function
    # sgd -> stochastic gradient descent
    MLP_classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=[100], activation='logistic', solver='sgd')
    MLP_classifier.fit(x_train, y_train)
    MLP_metrics = get_performance_metrics(MLP_classifier, x_test, y_test)

    # 6.f
    # activation: sigmoid, tang, relu, identity
    # network architecture: [30, 50] OR [10, 10, 10]
    # solver: sgd or adam(optimized stochastic gradient descent)
    top_MLP_classifier = sklearn.neural_network.MLPClassifier([10, 10, 10], activation='relu', solver='adam')
    top_MLP_classifier.fit(x_train, y_train)
    top_MLP_metrics = get_performance_metrics(top_MLP_classifier, x_test, y_test)
    print()
