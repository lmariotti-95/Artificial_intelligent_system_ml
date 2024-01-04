# Import delle librerie necessarie
from multiprocessing import get_all_start_methods
from shutil import register_unpack_format
from typing import ValuesView
import weakref
from numpy.random import f
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier

from joblib import dump, load
import os

class Metric:
    name = ""
    classes = None
    accuracy = None
    missclassification = None
    precision = None
    recall_score = None
    f1 = None
    conf_matrix = []
    
    fpr = []
    tpr = []
    auc = []
    
    def __init__(self, name):
        self.name = name
        pass
    
    def Get_metrics(self):
        return {
            "accuracy": self.accuracy,
            "missclassification": self.missclassification,
            "precision": self.precision,
            "recall_score": self.recall_score,
            "f1": self.f1,
        }
    
    def Print(self):
        print(self.name + ":")
        df = pd.DataFrame([self.Get_metrics()])
        print(df)
        print("-----------------")
        print("\n")
       
    def Plot_confusion_matrix(self):
        if(len(self.conf_matrix) == 0):
            return
        
        heatmap = sns.heatmap(self.conf_matrix, annot=True, cmap="viridis")
        heatmap.set_title(f"{self.name} Confusion Matrix")
        plt.show()
        
    def Plot_roc_curve(self):
        if(len(self.fpr) == 0 or len(self.tpr) == 0):
            return
        
        plt.plot(self.fpr, self.tpr, label=f"AUC={self.auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{self.name} ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

#region MODEL_TESTING        
def Export_model(name, classifier):
    model_filename = f"{name}.joblib"
    if os.path.isfile(model_filename):
        os.remove(model_filename)
    dump(classifier, model_filename)

def Test_model(file_name, test_file, labels):
    """
    Run a with a given model and a test dataset
    
    Parameters
    ----------
    :param str file_name: .joblib file that store a pre-generated model.
    :param str test_file: .csv file that hold the test dataset.
    :param str labels: Class names used as target for ML algorithms
    
    Returns 
    ----------
    Returns computed metric reporting all measures
    rtype: Metric 
    """
    assert(os.path.isfile(file_name) == True), f"Unvalid model file {file_name}"
    assert(os.path.isfile(test_file) == True), f"Unvalid test file {test_file}"
    
    # Loading model from file
    model = load(file_name)
    
    # Example usage of the loaded model
    raw_data = pd.read_csv(test_file)
    expected = raw_data[labels]
    
    # Removing class column
    raw_data = raw_data.drop(columns=labels)
    predictions = model.predict(raw_data)
    
    metric = Metric(f"Results: {file_name}")
    metric.accuracy = accuracy_score(expected, predictions)
    metric.missclassification = 1 - metric.accuracy
    metric.precision = precision_score(expected, predictions, average = "weighted", zero_division=0)
    metric.recall_score = recall_score(expected, predictions, average = "weighted", zero_division=0)
    metric.f1 = f1_score(expected, predictions, average = "weighted", zero_division=0)
    metric.conf_matrix = confusion_matrix(expected, predictions)

    return metric
#endregion  

#region CLASSIFIERS
def Compute_metrics(classifier, predictions, x_test, y_test, metric, export = False):
    metric.classes = classifier.classes_    
    metric.accuracy = accuracy_score(y_test, predictions)
    metric.missclassification = 1 - metric.accuracy
    metric.precision = precision_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.recall_score = recall_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.f1 = f1_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.conf_matrix = confusion_matrix(y_test, predictions)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
    y_pred_proba = classifier.predict_proba(x_test)[::,1]
    metric.fpr, metric.tpr, _ = roc_curve(y_test, y_pred_proba)
    metric.auc = roc_auc_score(y_test, y_pred_proba)
    
    if(export):
        Export_model(metric.name, classifier)
    
def Get_classes_wgt(y_train):
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    # n_samples / (n_classes * np.bincount(y))
    class_cnt = Counter(y_train.values)
    wgt_c0 = len(y_train.values) / (2 * class_cnt[0])
    wgt_c1 = len(y_train.values) / (2 * class_cnt[1])
    
    class_weight={
        0: wgt_c0,
        1: wgt_c1,
    }
    
    # returns 1 if oversampling is enabled   
    return class_weight
    
def Use_decision_tree_classifier(x_train, x_test, y_train, y_test, metric, export = False):
    """
    Decision tree classifier
    """
    classifier = DecisionTreeClassifier(
        class_weight = Get_classes_wgt(y_train),
        random_state = 42
    )
    
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    Compute_metrics(classifier, predictions, x_test, y_test, metric, export)
    
def Use_knn_classifier(x_train, x_test, y_train, y_test, metric, export = False):
    """
    Nearest neighbors classifier
    """
    classifier = KNeighborsClassifier(
        n_neighbors=2
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    Compute_metrics(classifier, predictions, x_test, y_test, metric, export)

def Use_svm_classifier(x_train, x_test, y_train, y_test, metric, export = False):   
    """
    Support Vector Machine (SVM)
    """
    classifier = SVC(class_weight='balanced', probability=True)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    Compute_metrics(classifier, predictions, x_test, y_test, metric, export)
    
def Use_adaboost_classifier(x_train, x_test, y_train, y_test, metric, export = False):
    """
    Ensemble AdaBoost
    """

    weak_learner = DecisionTreeClassifier(
        class_weight = Get_classes_wgt(y_train),
        max_depth = 2,
        random_state = 42,
    )
    
    classifier = AdaBoostClassifier(estimator=weak_learner)
    
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    Compute_metrics(classifier, predictions, x_test, y_test, metric, export)
    
#endregion

def Are_valid_metrics(metric_list):
    """
    Verify if a given list of metrics is valid aka:
    1) Is not null
    2) Has at least 1 element
    3) Each element is type(Metric)
    """
    if(metric_list == None):
        return False
    
    if(len(metric_list) == 0):
        return False
    
    if(all(isinstance(element, Metric) for element in metric_list) == False):
        return False
    
    return True

def Plot_metrics(metrics):
    """
    Plot data for a list of metrics given in input
    """
    # Watchdog for arguments consistency
    if(Are_valid_metrics(metrics) == False):
        return
    
    # If I get here I'm sure at least metrics[0] exist 
    x_lables = metrics[0].Get_metrics().keys()
    x_values = np.arange(len(x_lables)) 
    
    width = .15     # Bar width
    x_offset = 0    # Offset over x axis
    for m in metrics:
        if(m.accuracy == None):
            continue
        
        y_values = m.Get_metrics().values()
        plt.bar(x_values + x_offset, y_values, width=width, label=m.name)
        
        # Replacing x axis numbers with metric attributes name
        x_lables = m.Get_metrics().keys()
        plt.xticks(x_values + (width / 2), x_lables) 
        
        # Moving offset to the right
        x_offset = x_offset + width
    
    plt.xlabel("Metrics") 
    plt.ylabel("Value") 
    plt.title("Metrics comparison") 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show() 

def Record_metrics(metrics, file_name):
    # Check for arguments consistency
    if(Are_valid_metrics(metrics) == False):
        return
    
    keys = []  
    values = []
    
    keys.append("Metrics")
    
    # If I get here I'm sure at least metrics[0] exist 
    values.append(metrics[0].Get_metrics().keys())

    for m in metrics:
        keys.append(m.name)    
        values.append(m.Get_metrics().values())

    data = dict(zip(keys, values))
    df = pd.DataFrame(data)
    df.to_csv(file_name, sep=';', decimal=',', index=False)
    
#region DATASET_MANAGEMENT
def Load_dataset(file_name, test_perc, class_label, sampling = None, plot=False):
    """
    Load the dataset from a .csv file 

    Parameters
    ----------
    :param str file_name: csv file that hold the complete dataset.
    :param float training_perc: (0, 1) Percentage of samples used for training
    :param str labels: Class names used as target for ML algorithms
    :param str sampling: default = None
        "oversampling" - Enable oversampling for compensate imbalanced classes
        "undersampleing" - Enable undersampleing for compensate imbalanced classes
        "None" - Classes are loaded "as it is"
    :param bool plot: Enable classes distribution plotting
    
    Returns 
    ----------
    splitting: list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    
    assert(os.path.isfile(file_name) == True), f"Unvalid dataset file {file_name}"

    raw_data = pd.read_csv(file_name)
    
    data = raw_data.drop(columns=class_label)
    target = raw_data[class_label]
    
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_perc, random_state=42)
    
    # Check for sampling condition
    if(sampling != None):
        if(sampling == "oversampling"):
            ros = RandomOverSampler(
                sampling_strategy = 'minority',
                random_state=42
            )
            x_train, y_train = ros.fit_resample(x_train, y_train)
        elif(sampling == "undersampling"):
            ros =  RandomUnderSampler(
                sampling_strategy = 'not minority',
                random_state=42
            )
            x_train, y_train = ros.fit_resample(x_train, y_train)

    # Plot the distribution of class values
    if (plot == True):
        cnt = Counter(target.values)
        y_values = cnt.values()
        x_values = cnt.keys()

        plt.bar(x_values, y_values, width = .25)
        plt.title(f'Class: {class_label}')
        plt.ylabel('Count')
        plt.xlabel(class_label)

        plt.show()
        
    return (x_train, x_test, y_train, y_test)
#endregion

def main(verbose = False):
    class_label = "Bankrupt?"
    #dataset_filename = "data_set\\Company_Bankruptcy_Prediction.csv"
    dataset_filename = "data_set\\processed.csv"
    
    plot_metrics = True
    save_result = True
    dataset_sampling = "oversampling"

    test_size = .25
    
    if(verbose):
        print("Start training..")
        print(f"Sampling type: {dataset_sampling}")
        
    t_start = time()
    
    x_train, x_test, y_train, y_test = Load_dataset(
        dataset_filename, 
        test_size, 
        class_label, 
        sampling = dataset_sampling, 
        plot = False)
    
    if(verbose):
        print(f"Training done in: {time() - t_start}s")

    #Decision Tree
    if(verbose):
        print("Classifier: Decision Tree")
        
    dt = Metric("Decision Tree")
    Use_decision_tree_classifier(x_train, x_test, y_train, y_test, dt, export=False)
        
    #K-Nearest Neighbors (KNN)
    if(verbose):
        print("Classifier: K-Nearest Neighbors")
        
    knn = Metric("K-NN")
    Use_knn_classifier(x_train, x_test, y_train, y_test, knn, export=False)
    
    #SVM
    #if(verbose):
    #    print("Classifier: Support Vector Machine")
        
    #svm = Metric("SVM")
    #Use_svm_classifier(x_train, x_test, y_train, y_test, svm, export=False)

    if(verbose):
        print("Classifier: AdaBoost")
        
    ada = Metric("AdaBoost")
    Use_adaboost_classifier(x_train, x_test, y_train, y_test, ada, export=False)
    
    metric_list = []
    metric_list.append(dt)
    metric_list.append(knn)
    #metric_list.append(svm)
    metric_list.append(ada)
    
    """
    print("Test started")
    test_dataset_filename = "data_set\\Company_Bankruptcy_Prediction_1.csv"
    for m in metric_list:
        model_file_name = f"{m.name}.joblib"
        if(os.path.isfile(model_file_name)):
            test_metric = Test_model(model_file_name, test_dataset_filename, class_label)
            test_metric.Print()
    print("Test completed")
    """
    
    if(verbose):
        print(f"Process completed in: {time() - t_start}s")
        
        print(f"Training size: {1-test_size}")
        print(f"Testing size: {test_size}") 
        
        for m in metric_list:
            if(m != None):
                m.Print()

    if (plot_metrics):
        Plot_metrics(metric_list)
        
        for m in metric_list:
            if(m != None):
                m.Plot_confusion_matrix()
                m.Plot_roc_curve()

        
    # Save result to csv for some report statistics
    if (save_result):
        Record_metrics(metric_list, "data_set\\results.csv")

main(verbose = True)