# Import delle librerie necessarie
from multiprocessing import get_all_start_methods
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from collections import Counter

class Metric:
    name = ""
    classes = None
    accuracy = None
    missclassification = None
    precision = None
    recall_score = None
    f1 = None
    conf_matrix = None
    
    fpr = dict()
    tpr = dict()
    
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
        print("\n")
       
    def Plot_confusion_matrix(self):
        heatmap = sns.heatmap(self.conf_matrix, annot=True, cmap='viridis')
        heatmap.set_title(f"{self.name} Confusion Matrix")
        plt.show()
        
    def Plot_roc_curve(self):
        # Plotta la curva ROC per ogni classe
        #plt.figure(figsize=(8, 8))

        for i in range(len(self.classes)):
            plt.plot(self.fpr[i], self.tpr[i], lw=2, label=f'ROC curve (class {self.classes[i]})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        

def Use_decision_tree_classifier(x_train, x_test, y_train, y_test, metric):
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    
    metric.classes = classifier.classes_    
    metric.accuracy = accuracy_score(y_test, predictions)
    metric.missclassification = 1 - metric.accuracy
    metric.precision = precision_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.recall_score = recall_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.f1 = f1_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.conf_matrix = confusion_matrix(y_test, predictions)
    
    #y_score = classifier.predict_proba(x_test)
    #y_test_bin = label_binarize(y_test, classes=classifier.classes_)
    #
    ## Calcola la curva ROC per ogni classe
    #fpr = dict()
    #tpr = dict()
    #
    #for i in range(len(classifier.classes_)):
    #    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    #
    #metric.fpr = fpr
    #metric.tpr = tpr
    
def Use_knn_classifier(x_train, x_test, y_train, y_test, metric):
    classifier = KNeighborsClassifier()
    classifier.fit(x_train,y_train)
    predictions = classifier.predict(x_test)
    
    metric.classes = classifier.classes_    
    metric.accuracy = accuracy_score(y_test, predictions)
    metric.missclassification = 1 - metric.accuracy
    metric.precision = precision_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.recall_score = recall_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.f1 = f1_score(y_test, predictions, average = "weighted", zero_division=0)
    metric.conf_matrix = confusion_matrix(y_test, predictions)
    
    #y_score = classifier.predict_proba(x_test)
    #y_test_bin = label_binarize(y_test, classes=classifier.classes_)
    #
    ## Calcola la curva ROC per ogni classe
    #fpr = dict()
    #tpr = dict()
    #
    #for i in range(len(classifier.classes_)):
    #    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    #
    #metric.fpr = fpr
    #metric.tpr = tpr

"""
Verify if a given list of metrics is valid aka:
1) Is not null
2) Has at least 1 element
3) Each element is type(Metric)
"""
def Are_valid_metrics(metric_list):
    if(metric_list == None):
        return False
    
    if(len(metric_list) == 0):
        return False
    
    if(all(isinstance(element, Metric) for element in metric_list) == False):
        return False
    
    return True

"""
Given a series of metrics plot their data
Verbose = True to enable value printing on console
"""
def Plot_metrics(metrics, verbose = False):
    # Watchdog for arguments consistency
    if(Are_valid_metrics(metrics) == False):
        return
    
    # If I get here I'm sure at least metrics[0] exist 
    x_lables = metrics[0].Get_metrics().keys()
    x_values = np.arange(len(x_lables)) 
    
    width = .25     # Bar width
    x_offset = 0    # Offset over x axis
    for m in metrics:
        y_values = m.Get_metrics().values()
        plt.bar(x_values + x_offset, y_values, width=width, label=m.name)
        
        # Replacing x axis numbers with metric attributes name
        x_lables = m.Get_metrics().keys()
        plt.xticks(x_values + (width / 2), x_lables) 
        
        # Moving offset to the right
        x_offset = x_offset + width
        
        # Print on console if verbose is enabled
        if verbose == True:
            m.Print()
    
    plt.xlabel("Metrics") 
    plt.ylabel("Value") 
    plt.title("Metrics comparison") 
    plt.legend() 
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
def size_mb(docs):
    return sum(len(s.encode("utf-8")) for s in docs) / 1e6

"""
Load the dataset from a .csv file 

Parameters
----------
file_name: string
    csv file that hold the complete dataset.

training_perc: float (0, 1) 
    Percentage of samples used for training.
    
labels: list of strings
    Class names used as target for ML algorithms
"""
def Load_dataset(file_name, test_perc, class_label, verbose=False, plot=False):
    t_start = time()
    raw_data = pd.read_csv(file_name)
    
    data = raw_data.drop(columns=class_label)
    target = raw_data[class_label]
    
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_perc, random_state=42)
    
    t_duration = time() - t_start
    
    if (verbose == True):
        print(f"Input data size: {format(size_mb(file_name), '.6f')}MB")
        print(f"Training duration: {format(t_duration, '.4f')}s")
        print("\n")
    
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

def main():
    class_label = "Bankrupt?"
    dataset_filename = "data_set\Company_Bankruptcy_Prediction.csv"
    
    plot_metrics = True
    save_result = False

    test_size = .1
    
    dt_metric = []
    knn_metric = []
        
    x_train, x_test, y_train, y_test = Load_dataset(dataset_filename, test_size, class_label, verbose = False, plot = True)

    #Decision Tree
    dt = Metric("Decision Tree")
    Use_decision_tree_classifier(x_train, x_test, y_train, y_test, dt)
    dt_metric.append(dt)
        
    #K-Nearest Neighbors (KNN)
    knn = Metric("K-NN")
    Use_knn_classifier(x_train, x_test, y_train, y_test, knn)
    knn_metric.append(knn)

    if (plot_metrics):
        Plot_metrics([dt, knn])
        
        dt.Print()
        knn.Print()
        
        dt.Plot_confusion_matrix()
        knn.Plot_confusion_matrix()
        
        """   
        dt.Plot_roc_curve()
            
        knn.Plot_roc_curve()
        """   

    # Save result to csv for some report statistics
    if (save_result):
        Record_metrics([dt, knn], "data_set\results.csv")
         
    """
    if (len(test_size) > 1):
        plt.plot(test_size, [m.accuracy for m in dt_metric], label="accuracy")
        plt.plot(test_size, [m.missclassification for m in dt_metric], label="missclassification")
        plt.plot(test_size, [m.precision for m in dt_metric], label="precision")
        plt.plot(test_size, [m.recall_score for m in dt_metric], label="recall_score")
        plt.plot(test_size, [m.f1 for m in dt_metric], label="f1")
        plt.title("Decision Tree") 
        plt.xlabel("Test size") 
        plt.ylabel("Value") 
        plt.legend() 
        plt.show()
    
        plt.plot(test_size, [m.accuracy for m in knn_metric], label="accuracy")
        plt.plot(test_size, [m.missclassification for m in knn_metric], label="missclassification")
        plt.plot(test_size, [m.precision for m in knn_metric], label="precision")
        plt.plot(test_size, [m.recall_score for m in knn_metric], label="recall_score")
        plt.plot(test_size, [m.f1 for m in knn_metric], label="f1")
        plt.title("K-NN") 
        plt.xlabel("Test size") 
        plt.ylabel("Value") 
        plt.legend() 
        plt.show()
    """
    
main()