import pandas
import matplotlib.pyplot as plt
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
    set_of_sex_values = pandas.Categorical(csv_data['Sex'], ordered=False).unique().tolist()
    # ordered values; BP shares values with Cholesterol
    # reason for sorting is to make it consistent with PDF's order
    set_of_drug_values = pandas.Categorical(csv_data['Drug'], ordered=False).unique().tolist()
    set_of_drug_values.sort()
    set_of_bp_values = pandas.Categorical(csv_data['BP'], ordered=True, categories=['LOW', 'NORMAL', 'HIGH'])

    processed_data_frame = csv_data.replace(to_replace=set_of_sex_values, value=range(len(set_of_sex_values)))
    processed_data_frame = processed_data_frame.replace(to_replace=set_of_drug_values, value=range(len(set_of_drug_values)))
    processed_data_frame = processed_data_frame.replace(to_replace=set_of_bp_values.categories.tolist(), value=range(len(set_of_bp_values.categories.tolist())))
    return processed_data_frame, range(len(set_of_drug_values))


def prep_classifier_for_analysis():
    processed_dataset, classes_index = convert_to_numerical_format()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(processed_dataset, classes_index)
