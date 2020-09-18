import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix
from BaselineRemoval import BaselineRemoval
from scipy.signal import welch, find_peaks, savgol_filter

sns.set_style("whitegrid")
plt.style.use('ggplot')
sns.set(font_scale=1.5)
sns.set_palette("bright")


def load_data(train_folder, test_folder, train_labels, test_labels):
    '''
    loads train and test signals and labels
    input: train and test signal and label folder in .txt form
    output:numpy ndarray of train, test signals and labels
    '''
    train_pathlist = sorted(Path(train_folder).rglob('*.txt'))
    test_pathlist = sorted(Path(test_folder).rglob('*.txt'))

    train_signals = [np.loadtxt(path) for path in train_pathlist]
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    test_signals = [np.loadtxt(path) for path in test_pathlist]
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = np.loadtxt(Path(train_labels))
    test_labels = np.loadtxt(Path(test_labels))

    print("number of train signals, length of each signal, number of components:", train_signals.shape)
    print("number of test signals, length of each signal, number of components:", test_signals.shape)
    return train_signals, test_signals, train_labels, test_labels


def get_classification_results(models, X_train, X_test, y_train, y_test):
    
    '''
    inputs:
    models: Dictionary of ML models with Gridsearch parameters.
    output:
    Prints classification reports and confusion matrices.
    '''
    
    for i in models:
        models[i].fit(X_train, y_train)
        print(i, "Report:")
        print("############################################")
        print("Train Accuracy: {}".format(models[i].score(X_train, y_train)))
        print("Test Accuracy : {}".format(models[i].score(X_test, y_test)))
        y_pred = models[i].predict(X_test)
        best_params = models[i].best_params_
        best_params_df = pd.DataFrame(best_params, index=[0])
        print (" ")
        print("Best parameters:")
        display(best_params_df)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        print (" ")
        print("Classification Report:")
        display(df)
        cf_matrix = confusion_matrix(y_test, y_pred)
        print (" ")
        print("Confusion Matrix:")
        df1 = pd.DataFrame(cf_matrix)
        display(df1)
        cf_matrix_normalized_p = cf_matrix / cf_matrix.astype(np.float).sum(axis=0) 
        cf_matrix_normalized_r = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
        plt.figure(figsize=[8, 6])
        print("Normalized precision cf")
        sns.heatmap(cf_matrix_normalized_p, annot=True, cmap='Blues')
        plt.show()
        plt.figure(figsize=[8, 6])
        print("Normalized recall cf")
        sns.heatmap(cf_matrix_normalized_r, annot=True, cmap='Blues')
        plt.show()
        
        
def get_cnn_cf_matrix(model, X_test_tensor, test_labels):
    
    '''
    inputs:
    model: CNN model, test data in tensor form, test labels.
    output:
    Prints classification reports and confusion matrices.
    '''
    
    with torch.no_grad():
        output = model(X_test_tensor)
        pred = np.argmax(output, axis=1) + 1
        report = classification_report(pred, test_labels, output_dict=True)
        df = pd.DataFrame(report).transpose()
        display(df)
        cf_matrix = confusion_matrix(pred, test_labels)
        df1 = pd.DataFrame(cf_matrix)
        print("Number of samples in each class in test set:", np.unique(test_labels, return_counts=True))
        display(df1)
        plt.figure(figsize=[8, 6])
        cf_matrix_normalized_p = cf_matrix / cf_matrix.astype(np.float).sum(axis=0) 
        cf_matrix_normalized_r = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
        print("Normalized precision cf:")
        sns.heatmap(cf_matrix_normalized_p, annot=True, cmap='Blues')
        plt.show()
        plt.figure(figsize=[8, 6])
        print("Normalized recall cf:")
        sns.heatmap(cf_matrix_normalized_r, annot=True, cmap='Blues')
        plt.show()


def plot_stem (x, y, title):
    '''
    input: frequencies and intensities as lists (x, y). title is user defined title
    output: Stem plot
    
    '''
    plt.figure(figsize=[8, 6])
    plt.stem(x, y, use_line_collection=True)
    plt.xlabel('time/Sec')
    plt.ylabel('amplitude')
    plt.title(title)
    plt.plot(x, y)
    plt.show()


def plot_feature(x, y, feature):
    '''
    input: frequencies and intensities as lists (x, y). title is user defined title for features, FFT, PSD, etc.
    output: baseline smoothed frequency spectra with peak detection
    
    '''
    baseObj=BaselineRemoval(y)
    y=baseObj.ModPoly(2)
    y = savgol_filter(y, 5, 2)
    peaks, _ = find_peaks(y)
    first_n_freq = x[peaks][np.argsort(-y[peaks])][0:5]
    first_n_int =  y[peaks][np.argsort(-y[peaks])][0:5]
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks], "x")
    plt.xlabel('Freq/Hz')
    plt.ylabel('amplitude')
    plt.title(feature)
    plt.show()    






