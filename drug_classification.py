import pandas
import matplotlib.pyplot as plt


dataset_file = "./Documents/drug200.csv"


# T2Q2; Class will be 'Drug' types
def generate_pdf_of_instance_distribution():
    csv_data = pandas.read_csv(dataset_file)
    class_names = csv_data['Drug'].unique().tolist()
    class_names.sort()
    instance_number = csv_data.groupby('Drug').size().tolist()
    plt.plot(class_names, instance_number)
    plt.savefig("./Output/drug-distribution.pdf")


def prep_classifier_for_analysis():
    generate_pdf_of_instance_distribution()
