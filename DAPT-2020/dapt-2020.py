import numpy as np
import os.path
import random
import pandas as pd
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import csv
from csv import writer
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neural_network import MLPClassifier as mlp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#Data Pre-processing
training_file_name = "./train.csv"
testing_file_name = "./test.csv"

run_num=1

def load_data(file_name_train, file_name_test):
    '''
    This function loads and preprocess a training and test csv files
    :param file_name_train: Training csv file
    :param file_name_test: Testing csv file
    :return: X_train, X_test, y_train, y_test, column_names
    '''
    try:
        df_train = pd.read_csv(file_name_train, encoding="ISO-8859-1")
        df_test = pd.read_csv(file_name_test, encoding="ISO-8859-1")
    except FileNotFoundError:
        print(f"Error: File '{file_name_train}' not found.")
        print(f"Error: File '{file_name_test}' not found.")
        return None

    X_train = df_train.iloc[:, :-1].to_numpy()
    X_test = df_test.iloc[:, :-1].to_numpy()

    #Standerdizing the data using Standardscaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    column_names = df_train.columns[:-1].tolist()

    y_train = LabelEncoder().fit_transform(df_train.iloc[:, -1])
    y_test = LabelEncoder().fit_transform(df_test.iloc[:, -1])

    return X_train_scaled, X_test_scaled, y_train, y_test, column_names #With normalization

def get_seed_val(run_num):
    '''
    This function returns seed value for 30 runs on the basis of run_num
    The run_num will be taken from command line argument. run_num will be started from 1 in sgegrid.
    '''
    seed_val = [332, 70, 82, 684, 319, 292, 351, 634, 653, 451, 429, 168, 272, 225, 410, 654, 454, 164, 520, 427, 405, 602, 643, 296, 608, 199, 430, 504, 389, 306]
    return (seed_val[run_num - 1])

def append_list_as_row(file_name, list_of_elem):
    '''
    This function append the rows for the file file_name
    :param file_name: The csv file which rows need to be appended
    :param list_of_elem:  The items of the rows to append
    :return: appended rows
    '''
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def calculate_metrics(y_true, y_pred, train_time, pred_time):
    bal_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1_score_val = f1_score(y_true, y_pred, average='micro')

    total_tp = total_tn = total_fp = total_fn = 0
    for class_label in range(len(cm)):
        tp = sum((y_true == class_label) & (y_pred == class_label))
        tn = sum((y_true != class_label) & (y_pred != class_label))
        fp = sum((y_true != class_label) & (y_pred == class_label))
        fn = sum((y_true == class_label) & (y_pred != class_label))
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    combined_tnr = total_tn / (total_tn + total_fp)
    combined_fpr = 1 - combined_tnr

    return bal_acc, cm, f1_score_val, combined_tnr, combined_fpr, train_time, pred_time

def classifier_wrapper(classifier, X_train, X_test, y_train, y_test):
    start_time1 = datetime.datetime.now()
    classifier.fit(X_train, y_train)
    end_time1 = datetime.datetime.now()
    train_time = end_time1 - start_time1

    start_time2 = datetime.datetime.now()
    y_pred = classifier.predict(X_test)
    end_time2 = datetime.datetime.now()
    pred_time = end_time2 - start_time2
    return calculate_metrics(y_test, y_pred, train_time, pred_time)

def dt_classifier(X_train, X_test, y_train, y_test):
    classifier = dt(random_state=101)
    return classifier_wrapper(classifier, X_train, X_test, y_train, y_test)

def rf_classifier(X_train, X_test, y_train, y_test):
    classifier = rf(random_state=101)
    return classifier_wrapper(classifier, X_train, X_test, y_train, y_test)

def svm_classifier(X_train, X_test, y_train, y_test):
    classifier = SVC(random_state=101)
    return classifier_wrapper(classifier, X_train, X_test, y_train, y_test)

def gnb_classifier(X_train, X_test, y_train, y_test):
    classifier = gnb()
    return classifier_wrapper(classifier, X_train, X_test, y_train, y_test)

def knn_classifier(X_train, X_test, y_train, y_test):
    classifier = knn()
    return classifier_wrapper(classifier, X_train, X_test, y_train, y_test)

def mlp_classifier(X_train, X_test, y_train, y_test):
    bal_acc_list = []
    f1_score_list = []
    tnr_list = []
    fpr_list = []
    train_time_list = []
    pred_time_list = []
    for seed in range(1, 31):
        seed_int = get_seed_val(seed)
        classifier = mlp(random_state=seed_int)
        bal_acc, cm, f1_score_val, combined_tnr, combined_fpr, train_time, pred_time \
            = classifier_wrapper(classifier, X_train, X_test, y_train, y_test)
        bal_acc_list.append(bal_acc)
        f1_score_list.append(f1_score_val)
        tnr_list.append(combined_tnr)
        fpr_list.append(combined_fpr)
        train_time_list.append(train_time.total_seconds())  # Convert to seconds
        pred_time_list.append(pred_time.total_seconds())
    avg_bal_acc = np.mean(bal_acc_list)
    bal_acc_std = np.std(bal_acc_list)
    avg_f1_score = np.mean(f1_score_list)
    f1_score_std = np.std(f1_score_list)
    avg_tnr = np.mean(tnr_list)
    tnr_std = np.std(tnr_list)
    avg_fpr = np.mean(fpr_list)
    fpr_std = np.std(fpr_list)
    # Convert average times and std dev back to timedelta
    avg_train_time = datetime.timedelta(seconds=np.mean(train_time_list))
    train_time_std = datetime.timedelta(seconds=np.std(train_time_list))
    avg_pred_time = datetime.timedelta(seconds=np.mean(pred_time_list))
    pred_time_std = datetime.timedelta(seconds=np.std(pred_time_list))
    return (avg_bal_acc, bal_acc_std, avg_f1_score, f1_score_std, avg_tnr, tnr_std, avg_fpr, fpr_std,
            avg_train_time, train_time_std, avg_pred_time, pred_time_std)

def csv_result(run_num, seed_int,
               b_acc_dt, cm_dt, f1_score_dt, tnr_dt, fpr_dt,
               b_acc_rf, f1_score_rf, tnr_rf, fpr_rf,
               b_acc_svm, f1_score_svm, tnr_svm, fpr_svm,
               b_acc_gnb, f1_score_gnb, tnr_gnb, fpr_gnb,
               b_acc_knn, f1_score_knn, tnr_knn, fpr_knn,
               b_acc_mlp, b_acc_std_mlp, f1_score_mlp, f1_score_std_mlp, tnr_mlp, tnr_std_mlp, fpr_mlp, fpr_std_mlp,
               train_time_dt, pred_time_dt, train_time_rf, pred_time_rf, train_time_svm, pred_time_svm, train_time_gnb,
               pred_time_gnb, train_time_knn, pred_time_knn, train_time_mlp, train_time_std_mlp, pred_time_mlp,
               pred_time_std_mlp):
    if os.path.exists('Bal_Acc.csv'):
        print()
    else:
        with open('Bal_Acc.csv', 'a+', newline='') as f:
            header = ['run_num', 'seed_int',
'b_acc_dt', 'cm_dt', 'f1_score_dt', 'tnr_dt', 'fpr_dt',
'b_acc_rf', 'f1_score_rf', 'tnr_rf', 'fpr_rf',
'b_acc_svm', 'f1_score_svm', 'tnr_svm', 'fpr_svm',
'b_acc_gnb', 'f1_score_gnb', 'tnr_gnb', 'fpr_gnb',
'b_acc_knn', 'f1_score_knn', 'tnr_knn', 'fpr_knn',
'b_acc_mlp', 'b_acc_std_mlp', 'f1_score_mlp', 'f1_score_std_mlp', 'tnr_mlp', 'tnr_std_mlp', 'fpr_mlp', 'fpr_std_mlp',
'train_time_dt', 'pred_time_dt', 'train_time_rf', 'pred_time_rf', 'train_time_svm', 'pred_time_svm',
'train_time_gnb', 'pred_time_gnb','train_time_knn', 'pred_time_knn',
'train_time_mlp', 'train_time_std_mlp', 'pred_time_mlp', 'pred_time_std_mlp']
            filewriter = csv.DictWriter(f, fieldnames=header)
            filewriter.writeheader()
    row_contents = [str(run_num), str(seed_int),
str(b_acc_dt), str(cm_dt), str(f1_score_dt), str(tnr_dt), str(fpr_dt),
str(b_acc_rf), str(f1_score_rf), str(tnr_rf), str(fpr_rf),
str(b_acc_svm), str(f1_score_svm), str(tnr_svm), str(fpr_svm),
str(b_acc_gnb), str(f1_score_gnb), str(tnr_gnb), str(fpr_gnb),
str(b_acc_knn), str(f1_score_knn), str(tnr_knn), str(fpr_knn),
str(b_acc_mlp), str(b_acc_std_mlp), str(f1_score_mlp), str(f1_score_std_mlp), str(tnr_mlp), str(tnr_std_mlp), str(fpr_mlp), str(fpr_std_mlp),
str(train_time_dt), str(pred_time_dt), str(train_time_rf), str(pred_time_rf), str(train_time_svm), str(pred_time_svm),
str(train_time_gnb), str(pred_time_gnb), str(train_time_knn), str(pred_time_knn),
str(train_time_mlp), str(train_time_std_mlp), str(pred_time_mlp), str(pred_time_std_mlp)]
    # Append a list as new line to an old csv file
    append_list_as_row('Bal_Acc.csv', row_contents)

X_train, X_test, y_train, y_test, column_names = load_data(training_file_name, testing_file_name)
print(f"The shape of X_train is {X_train.shape}")
print(f"The shape of X_test is {X_test.shape}")
print(f"Number of instances per class in Training set: {np.unique(y_train)}")

if __name__ == "__main__":
    # Use random.seed to initialize same population for the specific seed value for any run
    seed_int = get_seed_val(run_num)
    print("The seed value for run " + str(run_num) + " is " + str(seed_int))
    # Starting Evolution using GP
    start_time = datetime.datetime.now()


    b_acc_dt, cm_dt, f1_score_dt, tnr_dt, fpr_dt, train_time_dt, pred_time_dt = \
        dt_classifier(X_train, X_test, y_train, y_test)
    b_acc_rf, cm_rf, f1_score_rf, tnr_rf, fpr_rf, train_time_rf, pred_time_rf = \
        rf_classifier(X_train, X_test, y_train, y_test)
    b_acc_svm, cm_svm, f1_score_svm, tnr_svm, fpr_svm, train_time_svm, pred_time_svm = \
        svm_classifier(X_train, X_test, y_train, y_test)
    b_acc_gnb, cm_gnb, f1_score_gnb, tnr_gnb, fpr_gnb, train_time_gnb, pred_time_gnb = \
        gnb_classifier(X_train, X_test, y_train, y_test)
    b_acc_knn, cm_knn, f1_score_knn, tnr_knn, fpr_knn, train_time_knn, pred_time_knn = \
        knn_classifier(X_train, X_test, y_train, y_test)
    (b_acc_mlp, b_acc_std_mlp, f1_score_mlp, f1_score_std_mlp, tnr_mlp, tnr_std_mlp, fpr_mlp, fpr_std_mlp,
     train_time_mlp, train_time_std_mlp, pred_time_mlp, pred_time_std_mlp) \
        = mlp_classifier(X_train, X_test, y_train, y_test)

    # Creating a CSV file containing all result.
    csv_result(run_num, seed_int,
               b_acc_dt, cm_dt, f1_score_dt, tnr_dt, fpr_dt,
               b_acc_rf, f1_score_rf, tnr_rf, fpr_rf,
               b_acc_svm, f1_score_svm, tnr_svm, fpr_svm,
               b_acc_gnb, f1_score_gnb, tnr_gnb, fpr_gnb,
               b_acc_knn, f1_score_knn, tnr_knn, fpr_knn,
               b_acc_mlp, b_acc_std_mlp, f1_score_mlp, f1_score_std_mlp, tnr_mlp, tnr_std_mlp, fpr_mlp, fpr_std_mlp,
               train_time_dt, pred_time_dt, train_time_rf, pred_time_rf, train_time_svm, pred_time_svm, train_time_gnb,
               pred_time_gnb, train_time_knn, pred_time_knn, train_time_mlp, train_time_std_mlp, pred_time_mlp,
               pred_time_std_mlp)

    # Calculating the time for each run
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print("Total time spent: ", total_time)
