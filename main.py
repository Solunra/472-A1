import text_classification as tc
import matplotlib
import sklearn

def main():
    split_test_set = tc.nb_classifier()
    # tc.generate_pdf_distribution_of_instance_distribution()
    tc.write_results_to_file()


# Always at the bottom of the file and will call main()
if __name__ == "__main__":
    main()
