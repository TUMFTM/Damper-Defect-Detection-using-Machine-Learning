import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import StratifiedKFold
import copy
import os
import utilities.data_handling as data_handling


def predict_auc(predictedOutlierProbability=None, true_labels=None):

    if isinstance(predictedOutlierProbability, list):
        auc = list()
        for single_predictedOutlierProbability in predictedOutlierProbability:
            fpr, tpr, thresholds = metrics.roc_curve(true_labels, single_predictedOutlierProbability)
            auc += [metrics.auc(fpr, tpr)]   # Calculate Area Under Curve of ROC
    else:
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictedOutlierProbability)
        auc = metrics.auc(fpr, tpr)
    return auc


def fit_classifier(featuresTraining, algorithm, featureSubsetNames=None, cvSplits=5, log_path_val=''):
    if featureSubsetNames is None:
        featureSubsetNames = list(featuresTraining.data.columns)

    if log_path_val is '':
        save_score_val = False
    else:
        save_score_val = True

    idx_val_list = list()
    trainedClassifier = list()
    validationAUC = list()
    idx_cv = StratifiedKFold(n_splits=cvSplits, shuffle=True, random_state=42)
    for cntCV, [idx_train, idx_val] in enumerate(idx_cv.split(featuresTraining, featuresTraining.labels)):
        featuresTrainingInlier, _ = data_handling.split_inlier_outlier(featuresTraining.iloc[idx_train])
        tmp_trainingdata = featuresTrainingInlier.data[featureSubsetNames]
        idx_val_list += [idx_val]

        tmp_algorithm = copy.deepcopy(algorithm)
        trainedClassifier += [tmp_algorithm.fit(tmp_trainingdata)]
        trainedClassifier[cntCV].featureNames = featureSubsetNames

        featuresValidation_singleCV = featuresTraining.iloc[idx_val_list[cntCV]]

        _, _, singleAUC, _ = test_classifier_on_test_data(featuresTesting=featuresValidation_singleCV, trainedClassifier=trainedClassifier[cntCV],
                scaler=None, nameTestset='val/cv'+str(cntCV), log_path=log_path_val, save_scores=save_score_val, transformInput=False)
        validationAUC += [singleAUC]
    print(validationAUC, 'AUC on validation data')

    return trainedClassifier, validationAUC


def predict_labels(trainedClassifier, featuresTesting):
    """Function for predicting on a test dataset"""

    if isinstance(trainedClassifier, list):
        y_pred_test = list()
        predictedOutlierProbability_Test = list()
        for singleClassifier in trainedClassifier:
            tmp_y_pred_test, tmp_predictedOutlierProbability_Test = predict_labels_for_one_classifier(singleClassifier, featuresTesting)
            y_pred_test += [tmp_y_pred_test]
            predictedOutlierProbability_Test += [tmp_predictedOutlierProbability_Test]
    else:
        y_pred_test, predictedOutlierProbability_Test = predict_labels_for_one_classifier(trainedClassifier, featuresTesting)

    return y_pred_test, predictedOutlierProbability_Test


def predict_labels_for_one_classifier(trainedClassifier, featuresTesting):

    # check if featureNames of classifier and testdata is equal
    if hasattr(trainedClassifier, 'featureNames'):
        if set(featuresTesting.data.columns) != set(trainedClassifier.featureNames):
            featuresTesting_data = featuresTesting.data.loc[:, trainedClassifier.featureNames]
        else:
            featuresTesting_data = featuresTesting.data
    else:
        print('Check of featureNames not possible, classifier has no specified feature list')
        featuresTesting_data = featuresTesting.data

    if hasattr(trainedClassifier, "predict_proba"):
        y_pred_test = trainedClassifier.predict(featuresTesting_data)
        if 0 in y_pred_test:
            y_pred_test[y_pred_test == 0] = -1
        predictedOutlierProbability_Test = trainedClassifier.predict_proba(featuresTesting_data)
        if predictedOutlierProbability_Test.ndim == 2:
            predictedOutlierProbability_Test = predictedOutlierProbability_Test[:, 1]
    elif hasattr(trainedClassifier, 'predict'):
        y_pred_test = trainedClassifier.predict(featuresTesting_data)
        predictedOutlierProbability_Test = None

    return y_pred_test, predictedOutlierProbability_Test


def evaluate_predictions(y_pred_test, predictedOutlierProbability_Test, featuresTesting):
    """Function for testing of a trained classifier on a testing dataset"""

    cm = metrics.confusion_matrix(featuresTesting.labels, y_pred_test)
    report = metrics.classification_report(featuresTesting.labels, y_pred_test)

    # Calculate Average Precision (which is the area under the precision-recall-curve)
    average_precision = metrics.average_precision_score(featuresTesting.labels, predictedOutlierProbability_Test)

    # Calculate Receiver Operating Characteristic
    fpr, tpr, thresholds = metrics.roc_curve(featuresTesting.labels, predictedOutlierProbability_Test)

    # Calculate Area Under Curve of ROC
    auc = metrics.auc(fpr, tpr)

    return cm, report, average_precision, auc, fpr, tpr


def test_classifier_on_test_data(featuresTesting=None, trainedClassifier=None, scaler=None, nameTestset='', log_path='', save_scores=False, transformInput=True):

    # transform test data by scaler
    if transformInput:
        featuresTesting.data = scaler.transform(featuresTesting.data)

    # Predict on test dataset
    pred_label, predictedOutlierProbability = predict_labels(
        trainedClassifier, featuresTesting)

    if isinstance(predictedOutlierProbability, list):
        auc = list()
        for cntCV, predictedOutlierProbability_cv in enumerate(predictedOutlierProbability):
            fpr, tpr, thresholds = metrics.roc_curve(featuresTesting.labels, predictedOutlierProbability_cv)
            auc += [metrics.auc(fpr, tpr)]

    else:
        fpr, tpr, thresholds = metrics.roc_curve(featuresTesting.labels, predictedOutlierProbability)
        auc = metrics.auc(fpr, tpr)

    if save_scores:
        singleRun = 0
        if not isinstance(predictedOutlierProbability, list):
            predictedOutlierProbability = [predictedOutlierProbability]
            singleRun = 1

        for cntCV, predictedOutlierProbability_cv in enumerate(predictedOutlierProbability):

                if singleRun==1:
                    addCVtoPath = ''
                else:
                    addCVtoPath = '/cv' + str(cntCV)

                # save prediction scores and labels for final test procedure
                if not os.path.isdir(log_path + '/result/' + nameTestset + addCVtoPath):
                    os.makedirs(log_path + '/result/' + nameTestset + addCVtoPath)
                np.savetxt(log_path + '/result/' + nameTestset + addCVtoPath + '/PredictionScores.csv',
                           predictedOutlierProbability_cv, fmt='%.4f', delimiter=",")
                np.savetxt(log_path + '/result/' + nameTestset + addCVtoPath + '/TrueLabels.csv',
                           featuresTesting.labels, fmt='%d', delimiter=",")
                np.savetxt(log_path + '/result/' + nameTestset + addCVtoPath + '/Index.csv',
                           featuresTesting.Prop.observationID.values,
                           fmt='%d', delimiter=",")

    return pred_label, predictedOutlierProbability, auc, featuresTesting.Prop.observationID.values


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax