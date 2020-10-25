import warnings
# ignore FutureWarnings needs to be on top of the script
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import copy
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tikzplotlib import save as tikz_save
from pyod.models import abod as pyodABOD
from pyod.models import knn as pyodKNN
from pyod.models import lof as pyodLOF
from pyod.models import lscp as pyodLSCP
from pyod.models.base import BaseDetector as pyodBase
from sklearn import metrics
from sklearn import preprocessing
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle

import Classifier.classifier_functions as classifier_functions
import utilities.data_handling as data_handling
import Featureanalyse.featureAnalysis as featureAnalysis


def main():

    # argument parser - not really used currently
    parser = argparse.ArgumentParser(description="script")
    # data parameters
    parser.add_argument("-logp", "--logspath", type=str,
                        default='results/' + datetime.now().strftime("%Y_%m_%d_%Hh_%M") + '/',
                        help="path for the logged data (avoid absolute path if possible)")
    config = parser.parse_args()

    ### Select location of mat-file containing the feature set
    ### BMW data
    # Manuelle Feature
    # config.pathFeatures = "../BMW/MachineLearning/ManuelleFeatures/200401_1025/Workspace.mat"  # DD2-F100
    # config.pathFeatures = "../BMW/MachineLearning/ManuelleFeatures/NoveltyDetection/2020_04_01_01h_08_manFeat/DatasizeAnalysis/1010_TrainingSamples.mat"  # DD2-F100

    # FFT Feature
    # config.pathFeatures = "../BMW/MachineLearning/FFT/200401_1023/Workspace.mat"
    # config.pathFeatures = "../BMW/MachineLearning/FFT/NoveltyDetection/2020_04_01_01h_10_FFT/DatasizeAnalysis/10099_TrainingSamples.mat"

    # Autoencoder Feature - trained on passive data
    # config.pathFeatures = "../BMW/RepresentationLearning/Autoencoder/200331_2324_trainedOnPassive/Workspace.mat"
    # config.pathFeatures = "../BMW/RepresentationLearning/Autoencoder/NoveltyDetection/2020_04_01_10h_16_AE/DatasizeAnalysis/10099_TrainingSamples.mat"

    # Sparsefilter Feature - trained on passive data
    # config.pathFeatures = "../BMW/RepresentationLearning/Sparsefilter/200330_2302_trainedOnPassive/Workspace.mat"
    config.pathFeatures = "../BMW/RepresentationLearning/Sparsefilter/NoveltyDetection/2020_04_01_01h_12_SF/DatasizeAnalysis/10099_TrainingSamples.mat"

    # config.saveFolder = '../BMW/MachineLearning/ManuelleFeatures/NoveltyDetection/'+datetime.now().strftime("%Y_%m_%d_%Hh_%M")
    # config.saveFolder = '../BMW/MachineLearning/ManuelleFeatures/NoveltyDetection/' + datetime.now().strftime("%Y_%m_%d_%Hh_%M")
    config.saveFolder = '../BMW/RepresentationLearning/Sparsefilter/NoveltyDetection/2020_04_01_01h_12_SF/DatasizeAnalysis/' + datetime.now().strftime("%Y_%m_%d_%Hh_%M")+'_ND_10099TrainingSamples'
    config.saveWorkspace = 1                    # save the workspace as *.pkl file to the config.saveFolder
    config.exportToMat = 1                      # export Outlier Scores and predicted class to mat-file
    config.exportToTikz = 0                     # export generated plots as *.tikz-file
    config.plot = 0                             # enable/disable all plots with this variable
    config.saveFig = 0                          # save generated plots as *.pdf
    config.plotEachAlgorithmForScaler = 0       # plot AUC value of all selected algorithms for a each scaler
    config.plotPrecisionRecallCurve = 0         # plot Precision-Recall-Curve of each configurated system
    config.plotROC = 0                          # plot Receiver-Operating-Characteristic of each configurated system
    config.plotEachScalerForAlgorithm = 0       # plot AUC value of all Scalers for each algorithm
    config.datasetToPlot = 'test'           # select, which testing dataset shall be used for the plots mentioned above
    config.nSamplesTestSize = {'test': 1, 'mass': 250, 'tire': 250}  # 500 # 125    # number of observations of each class (inlier/outlier) of the normal testing data in the test dataset
    config.nSamplesValidationSize = {'test': 0, 'mass': 0, 'tire': 0}         # number of observations of each class (inlier/outlier) of the normal testing data in the validation dataset
    config.loadMassData = 1
    config.loadTireData = 1
    config.nSamplesTestSizeMass = 250   #250    # number of observations of each class (inlier/outlier) of the mass data in the test dataset
    config.nSamplesTestSizeTire = 250 # 150     # number of observations of each class (inlier/outlier) of the tire data in the test dataset

    config.combineTrainingData = 0              # add inliers of mass and tire data to training data
    # config.combineTrainingDataAfterRFE = 1  # add inliers of mass and tire data to training data
    config.combineTrainingDataAfterRFE = 0  # add inliers of mass and tire data to training data
    config.combineTrainingDataset = ['tire']
    config.combineTestingData = 0               # add inliers and outliers of mass and tire data to testing data
    config.combineValidationDataRFE = 0
    # config.RFEevaluation = ['featureBlocks']
    # config.RFEevaluation = ['signalBlocks', 'featureBlocks', 'singleFeatures']    # perform Recursive Feature Elimination, select ['featureBlocks', 'signalBlocks', 'singleFeatures'] or '' or ['signalBlocks']
    config.RFEevaluation = ''
    config.RFEvalidationSet = 'test'        # select test dataset that is used as validation set for RFE
    config.trainClassifier = 1                  # train a single classifier or not
    config.number_CVsplits = 5

    # convert labels from dataset (e.g. 'RearAxleDamperMinus20Percent') to 1 for 'defect' and -1 for 'intact'
    config.conversionDict = {'passiveIntact': -1, 'intact': -1, 'allDampersDefect': 1, 'FLDamperDefect': 1, 'DamperDefect': 1,
                             'RRSpringPlus5_8PercStiffness': 1, 'RRSpringPlus16_4PercStiffness': 1,
                             'FLToeMinus21min': 1, 'FLToeMinus17min': 1,
                             'RRDamperDefect': 1, 'RearAxleDamperMinus20Percent': 1}

    # Define values of free variable to analyse (k for kNN Algorithms, nu for OCSVM)
    # both vectors shall have the same length
    # param1_vec_noOCSVM = np.linspace(2, 3, 2, dtype=int)
    # param1_vec_noOCSVM = np.array([2, 3, 5, 8, 10, 12, 15, 20, 35, 50])
    # param1_vec_OCSVM = np.array([0.001, 0.004, 0.006, 0.01, 0.02, 0.04, 0.06, 0.1, 0.2, 0.4])
    # param1_vec_noOCSVM = np.array([2, 3])
    param1_vec_noOCSVM = np.array([2])
    param1_vec_OCSVM = np.array([0.01])

    # set scalers for analysis by commenting in/out
    def allScalers():
        return {
            # "None": None,
            # "QuantileTransformerUniform": preprocessing.QuantileTransformer(output_distribution='uniform'),
            # "QuantileTransformerNormal": preprocessing.QuantileTransformer(output_distribution='normal'),
            "PowerTransformer": preprocessing.PowerTransformer(),
            # "RobustScaler": preprocessing.RobustScaler(),
            # "MaxAbsScaler": preprocessing.MaxAbsScaler(),
            # "MinMaxScaler": preprocessing.MinMaxScaler(),
            # "Normalizer": preprocessing.Normalizer(),
            # "StandardScaler": preprocessing.StandardScaler(),
        }

    # set algorithms for analysis by commeneting in/out (free variable e.g. n_neighbors is set later)
    def allAlgorithms():
        return {
                # "LOFminkowski": pyodLOF.LOF(metric='minkowski'),
                "LOFmanhatten": pyodLOF.LOF(metric='manhattan'),
                # "KNNmeanminkowski": pyodKNN.KNN(method='mean', metric='minkowski'),
                # "KNNlargestminkowski": pyodKNN.KNN(method='largest', metric='minkowski'),
                # "KNNmeamanhattan": pyodKNN.KNN(method='mean', metric='manhattan'),
                # "KNNlargestmanhattan": pyodKNN.KNN(method='largest', metric='manhattan'),
                # "ABODfast": pyodABOD.ABOD(method='fast'),
                # "ABODfull": pyodABOD.ABOD(method='default'),  # runs long time
                # "PCA_n_selected_comp": pyodPCA.PCA(),
                # "PCA_n_comp": pyodPCA.PCA(),
                # "AE": pyodAE.AutoEncoder(),
                # "OCSVMrbf": pyodOCSVM.OCSVM(kernel='rbf'),
                # "OCSVMlinear": pyodOCSVM.OCSVM(kernel='linear'),
                # "OCSVMpoly": pyodOCSVM.OCSVM(kernel='poly'),
                # "OCSVMsigmoid": pyodOCSVM.OCSVM(kernel='sigmoid'),
                # "IForest_nEstimators": pyodIForest.IForest(),
                # "IForest_maxFeatures": pyodIForest.IForest(),     # unten im Skript werden andere Parameter gesetzt
                # "CBLOF": pyodCBLOF.CBLOF(),       # läuft noch nicht
                # "LOCI": pyodLOCI.LOCI(),        # dauert sehr lange, erstmal auskommentiert
                # "FeatureBaggingLoOuFa": pyodFeatBagg.FeatureBagging(base_estimator=pyodLOF.LOF(metric='manhattan', n_neighbors=2)),       # bringt für LOF keine Verbesserung
                # "FeatureBaggingKNeNe": pyodFeatBagg.FeatureBagging(base_estimator=pyodKNN.KNN(method='mean', metric='manhattan')),        # bringt nichts für KNN
                # "HBOS": pyodHBOS.HBOS(),
                # "MCD": pyodMCD.MCD(),
                # "XGBoost": pyodXGB.XGBOD(),
                # "LSCP": 0,      # initialisierung muss direkt mit detector_list erfolgen -> später im Code initialisieren
                }

    # add abbreviation to folder name depending on the used features
    if "Sparsefilter" in config.pathFeatures:
        config.saveFolder = config.saveFolder + "_SF/"
    elif "Autoencoder" in config.pathFeatures:
        config.saveFolder = config.saveFolder + "_AE/"
    elif "FFT" in config.pathFeatures:
        config.saveFolder = config.saveFolder + "_FFT/"
    elif "ManuelleFeatures" in config.pathFeatures:
        config.saveFolder = config.saveFolder + "_manFeat/"
    else:
        config.saveFolder = config.saveFolder + "/"

    # make sure directory exists for writing the file
    if not os.path.isdir(config.saveFolder) and \
            (config.saveFig or config.exportToTikz or config.saveWorkspace or config.exportToMat):
        os.makedirs(config.saveFolder)

    # Load Training Data
    featuresTrainingOrig = data_handling.load_feature_set(config.pathFeatures, 'featuresTraining')
    featuresTrainingOrig['labels'] = data_handling.convert_labels(featuresTrainingOrig['labels'], config.conversionDict)
    featuresTrainingOrigInlier, featuresTrainingOrigOutlier = data_handling.split_inlier_outlier(featuresTrainingOrig)

    # balance inlier/outlier
    featuresTrainingOrigInlier = featuresTrainingOrigInlier[:min(featuresTrainingOrigInlier.shape[0], featuresTrainingOrigOutlier.shape[0])]
    featuresTrainingOrigOutlier = featuresTrainingOrigOutlier[:min(featuresTrainingOrigInlier.shape[0], featuresTrainingOrigOutlier.shape[0])]
    featuresTrainingUnscaled = featuresTrainingOrigInlier.append(featuresTrainingOrigOutlier)


    # Load Testing Data
    rawfeaturesTesting = dict()
    rawfeaturesTesting['test'] = data_handling.load_feature_set(config.pathFeatures, 'testdataTesting')

    # try to load 'mass' and 'tire' data if it exists
    if config.loadMassData:
        rawfeaturesTesting['mass'] = data_handling.load_feature_set(config.pathFeatures, 'testDD2Mass')
        if rawfeaturesTesting['mass'] is None: # delete 'None'-Type keys (e.g. 'mass' or 'tire')
            del rawfeaturesTesting['mass']
    if config.loadTireData:
        rawfeaturesTesting['tire'] = data_handling.load_feature_set(config.pathFeatures, 'testDD2Tire')
        if rawfeaturesTesting['tire'] is None: # delete 'None'-Type keys (e.g. 'mass' or 'tire')
            del rawfeaturesTesting['tire']

    featuresTestingUnscaled = dict()
    featuresValidationUnscaled = dict()
    tmpCombinedFeaturesTesting = pd.DataFrame()
    tmpCombinedFeaturesValidation = pd.DataFrame()

    # generate testing and training data
    for key in rawfeaturesTesting.keys():

        # convert labels to inliers (intact) and outliers (defective)
        rawfeaturesTesting[key]['labels'] = data_handling.convert_labels(rawfeaturesTesting[key]['labels'],
                                                                      config.conversionDict)
        inlier, outlier = data_handling.split_inlier_outlier(rawfeaturesTesting[key])   # split inlier/outlier

        # extract specified number of inlier and outlier for training, testing and validation
        if config.nSamplesTestSize[key] != 1:
            trainInlier, testInlier = train_test_split(inlier, test_size=config.nSamplesTestSize[key], random_state=42)
            trainOutlier, testOutlier = train_test_split(outlier, test_size=config.nSamplesTestSize[key], random_state=42)

            # balance inlier/outlier
            trainInlier = trainInlier[:min(trainInlier.shape[0], trainOutlier.shape[0])]
            trainOutlier = trainOutlier[:min(trainInlier.shape[0], trainOutlier.shape[0])]
        else:
            testInlier = shuffle(inlier, random_state=42)
            testOutlier = shuffle(outlier, random_state=42)

        # balance inlier/outlier
        testInlier = testInlier[:min(testInlier.shape[0], testOutlier.shape[0])]
        testOutlier = testOutlier[:min(testInlier.shape[0], testOutlier.shape[0])]

        # generate dataframe for unscaled testing data
        featuresTestingUnscaled[key] = testInlier
        featuresTestingUnscaled[key] = featuresTestingUnscaled[key].append(testOutlier)

        print(key, 'data inlier:  ', str(testInlier.shape[0]))
        print(key, 'data outlier: ', str(testOutlier.shape[0]))

        if config.combineTrainingData:
            if key in config.combineTrainingDataset:
                featuresTrainingUnscaled = featuresTrainingUnscaled.append(trainInlier)
                featuresTrainingUnscaled = featuresTrainingUnscaled.append(trainOutlier)


    # get scalers and algorithms from defined function
    scalers = allScalers()
    algorithms = allAlgorithms()

    # initialize some variables
    trainedClassifier = data_handling.initialize_multilevel_dict([scalers.keys(), algorithms.keys()])
    algorithmParameter = dict()
    predictedLabel = data_handling.initialize_multilevel_dict([featuresTestingUnscaled.keys(), scalers.keys(), algorithms.keys()])
    predictedOutlierProbability = data_handling.initialize_multilevel_dict([featuresTestingUnscaled.keys(), scalers.keys(), algorithms.keys()])
    AUC = data_handling.initialize_multilevel_dict([featuresTestingUnscaled.keys(), scalers.keys(), algorithms.keys()])
    observationID = data_handling.initialize_multilevel_dict([featuresTestingUnscaled.keys(), scalers.keys(), algorithms.keys()])
    resultTest = {'AUC': AUC, 'predictedLabel': predictedLabel, 'predictedOutlierProbability': predictedOutlierProbability, 'observationID': observationID}

    predictedLabelVal = data_handling.initialize_multilevel_dict([scalers.keys(), algorithms.keys()])
    predictedOutlierProbabilityVal = data_handling.initialize_multilevel_dict([scalers.keys(), algorithms.keys()])
    AUCVal = data_handling.initialize_multilevel_dict([scalers.keys(), algorithms.keys()])
    observationIDVal = data_handling.initialize_multilevel_dict([scalers.keys(), algorithms.keys()])
    resultVal = {'AUC': AUCVal, 'predictedLabel': predictedLabelVal, 'predictedOutlierProbability': predictedOutlierProbabilityVal, 'observationID': observationIDVal}

    # iterate across all selected scalers
    for idxScaler, (nameScaler, scaler) in enumerate(scalers.items()):

        # iterate across all selected algorithms
        for idxAlgorithm, (nameAlgorithm, algorithm) in enumerate(algorithms.items()):

            # select parameter vector for algorithm parameter
            if "OCSVM" in nameAlgorithm:
                param1_vec = param1_vec_OCSVM
            else:
                param1_vec = param1_vec_noOCSVM

            # iterate across all algorithm parameters
            for idxParam1, param1 in enumerate(param1_vec):

                nameParam1 = 'p'+str(param1)

                # set parameters of algorithm
                if "IForest_nEstimators" in nameAlgorithm:
                    algorithm.set_params(n_estimators=int(param1))
                elif "IForest_maxFeatures" in nameAlgorithm:
                    if param1 >= 1:
                        param1 /= max(param1_vec)
                    algorithm.set_params(max_features=param1)
                elif "PCA_n_selected_comp" in nameAlgorithm:
                    algorithm.set_params(n_selected_components=int(param1))
                elif "PCA_n_comp" in nameAlgorithm:
                    algorithm.set_params(n_components=int(param1))
                elif "HBOS" in nameAlgorithm:
                    algorithm.set_params(n_bins=int(param1))
                elif "LOCI" in nameAlgorithm:
                    algorithm.set_params(k=int(param1))
                elif "CBLOF" in nameAlgorithm:
                    algorithm.set_params(n_clusters=int(param1))
                elif "LOF" in nameAlgorithm:
                    algorithm.set_params(n_neighbors=int(param1))
                elif "KNN" in nameAlgorithm:
                    algorithm.set_params(n_neighbors=int(param1))
                elif "PCA" in nameAlgorithm:
                    algorithm.set_params(n_components=int(param1), svd_solver='full')
                elif "OCSVM" in nameAlgorithm:  # parameters for OCSVM need to be within 0...1
                    if param1 >= 1:
                        param1 /= max(param1_vec)
                        if param1 >= 1:
                            param1 = 0.99
                    algorithm.set_params(nu=param1)
                elif "ABOD" in nameAlgorithm:
                    algorithm.set_params(n_neighbors=int(param1))
                elif "AE" in nameAlgorithm:
                    algorithm.set_params(hidden_neurons=int(param1))
                elif "MCD" in nameAlgorithm:  # parameters for OCSVM need to be within 0...1
                    if param1 >= 1:
                        param1 /= max(param1_vec)
                        if param1 >= 1:
                            param1 = 0.99
                    algorithm.set_params(support_fraction=param1)
                elif "FeatureBagging" in nameAlgorithm:
                    algorithm.set_params(n_estimators=int(param1))
                elif "LSCP" in nameAlgorithm:
                    # copy list of algorithms to extra variable, because it doesn't work with trainedClassifier directly
                    algosLst = []
                    for cntAlgos in range(0, idxAlgorithm):  # 'idxAlgorithm' ist automatisch ausgeschlossen in Python
                        if isinstance(trainedClassifier[idxParam1][cntAlgos][idxScaler], pyodBase):
                            algosLst.append(trainedClassifier[idxParam1][cntAlgos][idxScaler])
                    algorithm = pyodLSCP.LSCP(detector_list=algosLst)
                elif "XGBoost" in nameAlgorithm:
                    # copy list of algorithms to extra variable, because it doesn't work with trainedClassifier directly
                    algosLst = []
                    for cntAlgos in range(0, idxAlgorithm):  # 'idxAlgorithm' ist automatisch ausgeschlossen in Python
                        algosLst.append(trainedClassifier[idxParam1][cntAlgos][idxScaler])
                    algorithm.set_params(estimator_list=algosLst)

                algorithmParameter[nameAlgorithm] = param1    # save set parameter

                # copy data to keep unscaled data untouched
                featuresTesting = copy.deepcopy(featuresTestingUnscaled)
                featuresTraining = copy.deepcopy(featuresTrainingUnscaled)
                featuresTrainingInlier, featuresTrainingOutlier = data_handling.split_inlier_outlier(featuresTraining)

                # fit scaler
                if scaler is not None:
                    scaler.fit(featuresTrainingInlier.data)
                    featuresTraining.data = scaler.transform(featuresTraining.data)

                if config.trainClassifier:
                    print('Starting classifier fitting using', nameScaler, nameAlgorithm, 'with x = ', str(param1))

                    if "XGBoost" in nameAlgorithm:
                        # some additional steps required for XGBoost classifier
                        featuresTrainingXGBoost = copy.deepcopy(featuresTraining)
                        # featuresTrainingXGBoost.add_to_dataset(data=featuresTrainingOutlier.data, labels=featuresTrainingOutlier.labels)
                        featuresTrainingXGBoost.labels[
                            featuresTrainingXGBoost.labels == -1] = 0  # convert labels because xgboost wants it
                        trainedClassifier[nameScaler][nameAlgorithm] = algorithm.fit(featuresTrainingXGBoost.data,
                                                                                              featuresTrainingXGBoost.labels)
                        featuresTrainingXGBoost.labels[featuresTrainingXGBoost.labels == 0] = -1  # re-convert labels
                    else:
                        trainedClassifier[nameScaler][nameAlgorithm], valAUC = classifier_functions.fit_classifier(
                            featuresTraining, algorithm, cvSplits=config.number_CVsplits, log_path_val=config.saveFolder)
                        print('Starting classifier fitting...finished')

                    # test classifier on testing data
                    for key in featuresTesting.keys():
                        resultTest['predictedLabel'][key][nameScaler][nameAlgorithm], \
                        resultTest['predictedOutlierProbability'][key][nameScaler][nameAlgorithm], \
                        resultTest['AUC'][key][nameScaler][nameAlgorithm], \
                        resultTest['observationID'][key][nameScaler][nameAlgorithm]= \
                            classifier_functions.test_classifier_on_test_data(featuresTesting=featuresTesting[key],
                                                  trainedClassifier=trainedClassifier[nameScaler][nameAlgorithm],
                                                                              scaler=scaler, nameTestset=key,
                                                                              log_path=config.saveFolder, save_scores=True)
                        print(resultTest['AUC'][key][nameScaler][nameAlgorithm], 'AUC on', key, 'data')

                # Run RFE
                if bool(config.RFEevaluation):

                    # backup initial path for saving the data
                    saveFolderOld = config.saveFolder

                    config.saveFolder = config.saveFolder + nameAlgorithm + '_' + nameScaler + '_' + nameParam1 + '/'

                    if not os.path.isdir(config.saveFolder):
                        os.makedirs(config.saveFolder)  # make sure directory exists for writing the file

                    featuresTrainingRFE = copy.deepcopy(featuresTrainingUnscaled)
                    featuresTrainingRFE.data = scaler.transform(featuresTrainingRFE.data)

                    RFE = dict()
                    for analysisMode in config.RFEevaluation:
                        print('Starting', analysisMode, 'RFE')
                        RFE[analysisMode] = featureAnalysis.perform_RFEusingAUC(featuresTrainingRFE, algorithm,
                                       config=config, analysisMode=analysisMode, nameAlgorithm=nameAlgorithm,  # featuresValidationRFE,
                                       nameScaler=nameScaler, paramAlgorithm=param1)

                        featuresTestingRFE = copy.deepcopy(featuresTestingUnscaled)
                        for key in featuresTestingRFE.keys():
                            featuresTestingRFE[key].data = scaler.transform(featuresTestingRFE[key].data)
                            RFE[analysisMode][key] = dict()
                            RFE[analysisMode][key]['AUC_raw'], RFE[analysisMode][key]['AUC_mean'], RFE[analysisMode][key]['AUC_std'], RFE[analysisMode][key]['numFeatures'] = featureAnalysis.testClassifierForEachFeatureSubset(
                                featuresTesting=featuresTestingRFE[key],
                                trainedClassifierPathLst=RFE[analysisMode]['trainedClassifierPath'])

                        # save predictions for best (based on validation data) RFE classifier
                        tmp_savepath = config.saveFolder + 'RFE_' + analysisMode + '_max_AUC'
                        idxOfmaxAUCofRFE = np.argmax(RFE[analysisMode]['AUC'])
                        trainedClassifier, valAUC = classifier_functions.fit_classifier(featuresTrainingRFE,
                                                                                        algorithm,
                                                                                        featureSubsetNames=
                                                                                        RFE[analysisMode][
                                                                                            'featureSubsetNames'][
                                                                                            idxOfmaxAUCofRFE],
                                                                                        cvSplits=config.number_CVsplits,
                                                                                        log_path_val=tmp_savepath)
                        for key in featuresTestingRFE.keys():
                            classifier_functions.test_classifier_on_test_data(featuresTesting=featuresTestingRFE[key],
                                                                              trainedClassifier=trainedClassifier,
                                                                              scaler=None, nameTestset=key,
                                                                              log_path=tmp_savepath,
                                                                              save_scores=True, transformInput=False)

                        # train with combined training data
                        tmp_savepath = config.saveFolder + 'RFE_' + analysisMode + '_max_AUC_combTrainData'
                        if config.combineTrainingDataAfterRFE:
                            featuresTrainingCombinedAfterRFE = copy.deepcopy(featuresTrainingUnscaled)
                            for key in rawfeaturesTesting.keys():
                                if key in config.combineTrainingDataset:
                                    inlier, outlier = data_handling.split_inlier_outlier(rawfeaturesTesting[key])  # split inlier/outlier

                                    # extract specified number of inlier and outlier for training, testing and validation
                                    trainInlier, _ = train_test_split(inlier, test_size=config.nSamplesTestSize[key],
                                                                               random_state=42)
                                    trainOutlier, _ = train_test_split(outlier,test_size=config.nSamplesTestSize[key],
                                                                                 random_state=42)
                                    # balance inlier/outlier
                                    trainInlier = trainInlier[:min(trainInlier.shape[0], trainOutlier.shape[0])]
                                    trainOutlier = trainOutlier[:min(trainInlier.shape[0], trainOutlier.shape[0])]

                                    featuresTrainingCombinedAfterRFE = featuresTrainingCombinedAfterRFE.append(trainInlier)
                                    featuresTrainingCombinedAfterRFE = featuresTrainingCombinedAfterRFE.append(trainOutlier)

                            # save predictions for best (based on validation data) RFE classifier
                            featuresTrainingCombinedAfterRFE.data = scaler.transform(featuresTrainingCombinedAfterRFE.data)
                            idxOfmaxAUCofRFE = np.argmax(RFE[analysisMode]['AUC'])
                            trainedClassifier, valAUC = classifier_functions.fit_classifier(featuresTrainingCombinedAfterRFE,
                                        algorithm, featureSubsetNames=RFE[analysisMode]['featureSubsetNames'][idxOfmaxAUCofRFE],
                                        cvSplits=config.number_CVsplits,log_path_val=tmp_savepath)

                            for key in featuresTestingRFE.keys():
                                classifier_functions.test_classifier_on_test_data(featuresTesting=featuresTestingRFE[key],
                                    trainedClassifier=trainedClassifier,scaler=None, nameTestset=key,log_path=tmp_savepath,
                                    save_scores=True, transformInput=False)

                    # save results
                    data_handling.save_workspace(config.saveFolder + '/RFE_result.pkl', RFE)
                    savemat(config.saveFolder + '/RFE_result.mat', {'RFE': RFE}, long_field_names=True, oned_as='column')

                    # restore initial path for saving the data; should be the last action of the RFE
                    config.saveFolder = saveFolderOld

    if config.saveWorkspace:
        # save workspace
        print('Saving workspace to ', config.saveFolder)
        data_handling.save_workspace(config.saveFolder + '/workspace_finalResult.pkl', locals())

    # if config.exportToMat:
        # save prediction results
        # savemat(config.saveFolder + '/result.mat', {'AUC': AUC, 'predictedLabel': predictedLabel, 'predictedOutlierProbability': predictedOutlierProbability}, long_field_names=True, oned_as='column')
        # savemat(config.saveFolder + '/result.mat', {'resultTest': resultTest, 'resultVal': resultVal}, long_field_names=True, oned_as='column')

        # # save feature array
        # for key in featuresTesting.keys():
        #     data_handling.save_to_mat(featuresTesting[key], config.saveFolder + '/featuresTesting_'+key+'.mat', 'featuresTesting'+key)

        # for key in featuresValidation.keys():
        #     data_handling.save_to_mat(featuresValidation[key], config.saveFolder + '/featuresValidation_'+key+'.mat', 'featuresValidation'+key)


    return 0


if __name__ == '__main__':
    main()
