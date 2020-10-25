import copy
import numpy as np
import Classifier.classifier_functions as classifier_functions
from sklearn import metrics
from utilities.data_handling import make_dataset as make_dataset
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import scipy.io
import utilities.data_handling as data_handling
from sklearn.model_selection import KFold


def RFE_inner_run(cntInnerNumFeatures=None, cntNumFeatures=None, featuresTrainingOuter=None,
                  listOfRemainingSingleFeatures=None,
                  listOfRemainingFeatureNamesToBeEvaluated=None, algorithm=None):  # featuresTestingOuter=None,

    innerSingleFeaturesToBeDeleted = [s for s in listOfRemainingSingleFeatures if listOfRemainingFeatureNamesToBeEvaluated[cntInnerNumFeatures] == s]
    if not innerSingleFeaturesToBeDeleted:
        innerSingleFeaturesToBeDeleted = [s for s in listOfRemainingSingleFeatures if listOfRemainingFeatureNamesToBeEvaluated[cntInnerNumFeatures] in s]

    tmpfeaturesTrainingInner = featuresTrainingOuter.drop(innerSingleFeaturesToBeDeleted, axis=1, level=1)

    trainedClassifier, auc = classifier_functions.fit_classifier(tmpfeaturesTrainingInner, algorithm)

    print('Outer run ', cntNumFeatures, '- Inner run ', cntInnerNumFeatures, '- AUC:', np.mean(auc), '+-', np.std(auc))

    return auc, trainedClassifier, cntInnerNumFeatures


def perform_RFEusingAUC(featuresTraining=None, algorithm=None, config=None, analysisMode=None,  # featuresValidation=None,
                        nameAlgorithm=None, nameScaler=None, paramAlgorithm=None):

    # extract parts of feature names to be analyzed according to analysisMode
    if 'featureBlocks' in analysisMode:
        listOfAllFeatureNamesToBeEvaluated = list(featuresTraining.data.columns)
        featureBlockNames = [i.split('_',1)[0] for i in listOfAllFeatureNamesToBeEvaluated]
        listOfAllFeatureNamesToBeEvaluated = list(sorted(set(featureBlockNames)))
        listOfAllFeatureNamesToBeEvaluated = [i + '_' for i in listOfAllFeatureNamesToBeEvaluated] # add underscore to prevent mixing up features of similar names

    elif 'signalBlocks' in analysisMode:
        allFeatNames = list(featuresTraining.data.columns)

        allFeatNamesSplitted = [i.split('_') for i in allFeatNames]
        uniqueEndingNumbers = sorted(set([i[-1] for i in allFeatNamesSplitted]))
        uniqueFeatNamesWoNumbers = sorted(set(['_'.join(i[0:-1]) for i in allFeatNamesSplitted]))

        if len(allFeatNames) / len(uniqueEndingNumbers) == len(uniqueFeatNamesWoNumbers):
            signalNames = uniqueFeatNamesWoNumbers
        else:
            featureBlockNames = [i.split('_', 1)[0] for i in allFeatNames]
            featureBlockNames = [i + '_' for i in featureBlockNames]

            numberOfFeaturesPerFeatureBlock = dict([x, featureBlockNames.count(x)] for x in set(featureBlockNames))
            featureBlockWithLeastFeatures = min(numberOfFeaturesPerFeatureBlock,
                                                key=numberOfFeaturesPerFeatureBlock.get)

            # extract all feature names of feature block with least features and delete last number with underscore
            featureNamesOfFeatureBlockWithLeastFeatures = ['_'.join(i.split('_')[0:-1]) for i in allFeatNames if
                                                           featureBlockWithLeastFeatures in i]

            # get feature names by splitting based on the string in featureBlockWithLeastFeatures
            signalNames = [i.split(featureBlockWithLeastFeatures, 1)[1] for i in
                           featureNamesOfFeatureBlockWithLeastFeatures]

        listOfAllFeatureNamesToBeEvaluated = list(signalNames)

    elif 'singleFeatures' in analysisMode:
        listOfAllFeatureNamesToBeEvaluated = list(featuresTraining.data.columns)

    numFeatures = len(listOfAllFeatureNamesToBeEvaluated)

    listOfRemainingSingleFeatures = featuresTraining.data.columns
    listOfRemainingFeatureNamesToBeEvaluated = listOfAllFeatureNamesToBeEvaluated

    RFE = dict()
    RFE['featuresSortedByRank'] = [None] * numFeatures
    RFE['featureSubsetNames'] = [None] * numFeatures
    RFE['trainedClassifierPath'] = [None] * numFeatures
    RFE['AUC'] = [None] * numFeatures
    RFE['AUC_std'] = [None] * numFeatures
    RFE['AUC_raw'] = [None] * numFeatures

    # calculation with all features
    tmpTrainedClassifier, valAUC = classifier_functions.fit_classifier(featuresTraining, algorithm)

    RFE['trainedClassifierPath'][numFeatures-1] = config.saveFolder + '/RFE_classifier/' + analysisMode + '/' + str(
        numFeatures - 1) + '.pkl'
    data_handling.save_classifier(RFE['trainedClassifierPath'][numFeatures-1], tmpTrainedClassifier)

    RFE['featureSubsetNames'][numFeatures-1] = list(featuresTraining.data.columns)

    RFE['AUC_raw'][numFeatures-1] = valAUC
    RFE['AUC'][numFeatures - 1] = np.mean(valAUC)
    RFE['AUC_std'][numFeatures - 1] = np.std(valAUC)

    featuresTrainingOuter = copy.deepcopy(featuresTraining)

    for cntNumFeatures in range(numFeatures-2, -1, -1):

        tmpTrainedClassifier = [None] * (cntNumFeatures+1)
        auc_mean = np.zeros((cntNumFeatures+1, 1))
        auc_std = np.zeros((cntNumFeatures+1, 1))
        auc = [None]*(cntNumFeatures + 1)

        func = partial(RFE_inner_run, listOfRemainingSingleFeatures=listOfRemainingSingleFeatures,
                       listOfRemainingFeatureNamesToBeEvaluated=listOfRemainingFeatureNamesToBeEvaluated,
                       cntNumFeatures=cntNumFeatures, featuresTrainingOuter=featuresTrainingOuter,
                       algorithm=algorithm)  # featuresTestingOuter=featuresTestingOuter,

        # use parallel computation
        with concurrent.futures.ProcessPoolExecutor() as executor: #ThreadPoolExecutor ProcessPoolExecutor
            for aucInner, trainedClassifierInner, cntInnerNumFeatures in executor.map(func, range(0, cntNumFeatures+1)):
                tmpTrainedClassifier[cntInnerNumFeatures] = trainedClassifierInner
                auc[cntInnerNumFeatures] = aucInner
                auc_mean[cntInnerNumFeatures,0] = np.mean(auc[cntInnerNumFeatures])
                auc_std[cntInnerNumFeatures, 0] = np.std(auc[cntInnerNumFeatures])

        idxmaxAUC = np.argmax(auc_mean)

        innerSingleFeaturesToBeDeleted = [s for s in listOfRemainingSingleFeatures if listOfRemainingFeatureNamesToBeEvaluated[idxmaxAUC] == s]
        if not innerSingleFeaturesToBeDeleted:
            innerSingleFeaturesToBeDeleted = [s for s in listOfRemainingSingleFeatures if listOfRemainingFeatureNamesToBeEvaluated[idxmaxAUC] in s]

        featuresTrainingOuter = featuresTrainingOuter.drop(innerSingleFeaturesToBeDeleted, axis=1, level=1)

        RFE['AUC'][cntNumFeatures] = auc_mean[idxmaxAUC, 0]
        RFE['AUC_std'][cntNumFeatures] = auc_std[idxmaxAUC, 0]
        RFE['AUC_raw'][cntNumFeatures] = auc[idxmaxAUC]
        RFE['featuresSortedByRank'][cntNumFeatures+1] = listOfRemainingFeatureNamesToBeEvaluated[idxmaxAUC]
        RFE['trainedClassifierPath'][cntNumFeatures] = config.saveFolder + '/RFE_classifier/'+analysisMode+'/'+str(cntNumFeatures)+'.pkl'
        data_handling.save_classifier(RFE['trainedClassifierPath'][cntNumFeatures], tmpTrainedClassifier[idxmaxAUC])
        if isinstance(tmpTrainedClassifier[idxmaxAUC], list):
            RFE['featureSubsetNames'][cntNumFeatures] = tmpTrainedClassifier[idxmaxAUC][0].featureNames
        else:
            RFE['featureSubsetNames'][cntNumFeatures] = tmpTrainedClassifier[idxmaxAUC].featureNames

        listOfRemainingSingleFeatures = RFE['featureSubsetNames'][cntNumFeatures]
        listOfRemainingFeatureNamesToBeEvaluated.remove(RFE['featuresSortedByRank'][cntNumFeatures+1])

        print('Max AUC:', auc_mean[idxmaxAUC], '+-', auc_std[idxmaxAUC], '- Deleted Feature:', RFE['featuresSortedByRank'][cntNumFeatures+1])

    RFE['featuresSortedByRank'][0] = listOfRemainingFeatureNamesToBeEvaluated[0]

    if config.plot:
        plt.figure()
        plt.errorbar(range(1, numFeatures+1, 1), RFE['AUC'], yerr=RFE['AUC_std'])
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.title('RFE'+nameAlgorithm+'Param:'+str(paramAlgorithm)+nameScaler)
        plt.legend()
        if config.saveFig:
            plt.savefig(config.saveFolder + 'RFE_'+nameAlgorithm+'_Param'+str(paramAlgorithm)+'_'+nameScaler+'.pdf')
        if config.exportToTikz:
            tikz_save(config.saveFolder + 'RFE_'+nameAlgorithm+'_Param'+str(paramAlgorithm)+'_'+nameScaler+'.tikz', encoding='utf8',
                      show_info=False)

    scipy.io.savemat(config.saveFolder + '/RFE_'+nameAlgorithm+'_'+nameScaler+'_'+str(paramAlgorithm)+'.mat', {'RFE': RFE}, long_field_names=True, oned_as='column')

    return RFE


def predictSingleClassifier(cntFeatureSet=None, featuresTesting=None, trainedClassifierPathLst=None):

    trainedClassifier = data_handling.load_classifier(trainedClassifierPathLst[cntFeatureSet])
    y_pred_test, predictedOutlierProbability_Test = classifier_functions.predict_labels(trainedClassifier=trainedClassifier, featuresTesting=featuresTesting)

    auc = classifier_functions.predict_auc(predictedOutlierProbability_Test, featuresTesting.labels)

    return auc, cntFeatureSet


def testClassifierForEachFeatureSubset(featuresTesting=None, trainedClassifierPathLst=None):

    numFeatureSets = len(trainedClassifierPathLst)
    auc_raw = [None]*numFeatureSets
    auc_mean = [None] * numFeatureSets
    auc_std = [None] * numFeatureSets

    func = partial(predictSingleClassifier, featuresTesting=featuresTesting, trainedClassifierPathLst=trainedClassifierPathLst)

    with concurrent.futures.ProcessPoolExecutor() as executor:  # ThreadPoolExecutor ProcessPoolExecutor
        for tmpauc, cntFeatureSet in executor.map(func, range(0, numFeatureSets)):
            auc_raw[cntFeatureSet] = tmpauc
            auc_mean[cntFeatureSet] = np.mean(tmpauc)
            auc_std[cntFeatureSet] = np.std(tmpauc)

    return auc_raw, auc_mean, auc_std, list(range(1, numFeatureSets+1))

