import pandas
import matplotlib.pyplot as plt
from sklearn import *
import sklearn
import numpy as np


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


def get_performance_metrics(classifier, x_test, y_test):
    metrics = {}
    predicted_y = classifier.predict(x_test)
    metrics['confusion_matrix'] = sklearn.metrics.confusion_matrix(y_test, predicted_y)
    metrics['classification_report'] = sklearn.metrics.classification_report(y_test, predicted_y, output_dict=True)
    metrics['accuracy_score'] = sklearn.metrics.accuracy_score(y_test, predicted_y)
    metrics['f1_score_macro'] = sklearn.metrics.f1_score(y_test, predicted_y, average='macro')
    metrics['f1_score_weighted'] = sklearn.metrics.f1_score(y_test, predicted_y, average='weighted')
    metrics['precision'] = sklearn.metrics.precision_score(y_test, predicted_y, average='macro')
    metrics['recall'] = sklearn.metrics.recall_score(y_test, predicted_y, average='macro')
    return metrics


def run_classifiers():
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
    top_MLP_classifier = sklearn.neural_network.MLPClassifier([30, 50], activation='tanh', solver='adam')
    top_MLP_classifier.fit(x_train, y_train)
    top_MLP_metrics = get_performance_metrics(top_MLP_classifier, x_test, y_test)

    # Used to get the best values for TOP_MLP (6.f); Uncomment to run
    # testing_best_parameters_for_top_mlp(x_train, y_train, x_test, y_test)

    return [nb_metrics, b_dt_metrics, t_dt_metrics, perceptron_metrics, MLP_metrics, top_MLP_metrics]


# For 7 and 8
def output_results():
    list_of_metrics = run_classifiers()
    all_results = [[[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]]
    # gets accuracy, f1 macro, f1 weighted into an array for calculation
    for i in range(10):
        current_metrics = run_classifiers()
        if i is 0:
            list_of_metrics = current_metrics
        for index, classifier_metric in enumerate(list_of_metrics):
            all_results[index][0].append(classifier_metric["accuracy_score"])
            all_results[index][1].append(classifier_metric["f1_score_macro"])
            all_results[index][2].append(classifier_metric["f1_score_weighted"])
    # average of all scores for each classifier
    all_averages = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    for classifier_index, classifier_scores in enumerate(all_results):
        for score_index, scores in enumerate(classifier_scores):
            all_averages[classifier_index][score_index] = np.average(scores)
    # standard deviation of all scores for each classifier
    all_std = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    for classifier_index, classifier_scores in enumerate(all_results):
        for score_index, scores in enumerate(classifier_scores):
            all_std[classifier_index][score_index] = np.std(scores)

    with open('./Output/drugs-performance.txt', 'w+') as file:
        order_of_metrics = ['NB', 'Base-DT', 'Top-DT with criterion: entropy, max_depth: [5, 10], min_samples_split: [2, 4, 6]',
                            'PER', 'Base-MLP', 'Top-MLP with activation: tanh, solver: adam, network arch.: [30, 50]']
        for index, classifier_metric in enumerate(list_of_metrics):
            file.write(f'a) {order_of_metrics[index]} ============================================\n')
            file.write(f'b) Confusion Matrix\n{classifier_metric["confusion_matrix"]}\n')
            classification_report = classifier_metric['classification_report']
            file.write(f'c)\n')
            for key_index, key in enumerate(classification_report):
                # only get the class values
                if key_index > 4:
                    break
                file.write(f'class {key}: \n')
                file.write(f'\t Precision: {str(classification_report[key]["precision"])}\n' +
                           f'\t Recall: {str(classification_report[key]["recall"])}\n' +
                           f'\t F1-Measure: {str(classification_report[key]["f1-score"])}\n')
            file.write(f'd) \n' +
                       f'\t Accuracy: {str(classifier_metric["accuracy_score"])}\n' +
                       f'\t Macro-Average F1-Score: {str(classifier_metric["f1_score_macro"])}\n' +
                       f'\t Weighted-Average F1-Score: {str(classifier_metric["f1_score_weighted"])}\n')
            file.write('\n\n')
        # end
        file.write('\n\n=== All Averages Per Class ===\n\n')
        for index, average_values in enumerate(all_averages):
            file.write(f'For classifier: {order_of_metrics[index]}\n')
            file.write(f'Average Accuracy: {str(average_values[0])}\n')
            file.write(f'Average Macro F1: {str(average_values[1])}\n')
            file.write(f'Average Weighted F1: {str(average_values[2])}\n\n')

        file.write('\n\n=== All Standard Deviation Per Class ===\n\n')
        for index, std_values in enumerate(all_std):
            file.write(f'For classifier: {order_of_metrics[index]}\n')
            file.write(f'Average Accuracy: {str(std_values[0])}\n')
            file.write(f'Average Macro F1: {str(std_values[1])}\n')
            file.write(f'Average Weighted F1: {str(std_values[2])}\n\n')


def testing_best_parameters_for_top_mlp(x_train, y_train, x_test, y_test):
    activation_values = ['logistic', 'tanh', 'relu', 'identity']
    network_architecture = [[30, 50], [10, 10, 10]]
    solver = ['sgd', 'adam']

    best_accuracy_score = 0
    best_accuracy_combo = []
    best_f1_score_macro = 0
    best_f1_score_macro_combo = []
    best_f1_score_weighted = 0
    best_f1_score_weighted_combo = []
    for activate in activation_values:
        for index, network in enumerate(network_architecture):
            for solve in solver:
                top_MLP_classifier = sklearn.neural_network.MLPClassifier(network, activation=activate, solver=solve)
                top_MLP_classifier.fit(x_train, y_train)
                top_MLP_metrics = get_performance_metrics(top_MLP_classifier, x_test, y_test)
                if top_MLP_metrics['accuracy_score'] > best_accuracy_score:
                    best_accuracy_score = top_MLP_metrics['accuracy_score']
                    best_accuracy_combo = [activate, index, solve]
                if top_MLP_metrics['f1_score_macro'] > best_f1_score_macro:
                    best_f1_score_macro = top_MLP_metrics['f1_score_macro']
                    best_f1_score_macro_combo = [activate, index, solve]
                if top_MLP_metrics['f1_score_weighted'] > best_f1_score_weighted:
                    best_f1_score_weighted = top_MLP_metrics['f1_score_weighted']
                    best_f1_score_weighted_combo = [activate, index, solve]
    print('best accuracy: ' + best_accuracy_combo[0] + ', [' + str(best_accuracy_combo[1]) + '], ' + best_accuracy_combo[2])
    print('best f1 score macro: ' + best_f1_score_macro_combo[0] + ', [' + str(best_f1_score_macro_combo[1]) + '], ' + best_f1_score_macro_combo[2])
    print('best f1 score weighted: ' + best_f1_score_weighted_combo[0] + ', [' + str(best_f1_score_weighted_combo[1]) + '], ' + best_f1_score_weighted_combo[2])
