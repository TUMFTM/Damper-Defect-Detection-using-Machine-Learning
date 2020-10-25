function classificationKNN = trainkNN(dataTraining, classifierOpts)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict


    % Convert input data to desired table format if it is a struct
    if isstruct(dataTraining)
        response = dataTraining.Label;
        dataTraining = dataTraining.data;
    else
        try
            response = dataTraining.Label;
        catch
            response = dataTraining.Label_ISOLATION;
        end
    end


    if istable(dataTraining)
        predictorNames = dataTraining.Properties.VariableNames';
        predictorNames(strcmp(predictorNames,'Label')) = [];   % get rid of fields "Label"
        predictorNames(strcmp(predictorNames,'Label_ISOLATION')) = [];   % get rid of fields "Label"
        predictors = dataTraining(:, predictorNames);
    else
        predictorNames = datastructTraining.featureNames;
        predictorNames = (~strcmp(datastructTraining.featureNames, 'Label')).* (~strcmp(datastructTraining.featureNames, 'Label_ISOLATION'));
        predictors = dataTraining(:, logical(predictorNames));
    end
    
%     predictorNames = dataTraining.Properties.VariableNames';
%     predictorNames(strcmp(predictorNames,'Label')) = [];   % get rid of fields "Label"
%     predictorNames(strcmp(predictorNames,'Label_ISOLATION')) = [];   % get rid of fields "Label"
%     predictors = dataTraining(:, predictorNames);

    if ~isfield(classifierOpts, 'ClassNames')
        classNames = unique(response);
    else
        classNames = classifierOpts.ClassNames;
    end

    % Create Partition for Holdout Validation
    if classifierOpts.validation <= 1
        cvPartition = cvpartition(response,'HoldOut',(1-classifierOpts.validation));
    else
        cvPartition = cvpartition(response,'kFold',classifierOpts.validation);
    end

    % Train a classifier
    fprintf('Classifier training...');
    tic
    classificationKNN = fitcknn(...
        predictors, ...
        response, ...
        'Distance', classifierOpts.Distance, ...
        'Exponent', classifierOpts.Exponent, ...
        'NumNeighbors', classifierOpts.NumNeighbors, ...
        'DistanceWeight', classifierOpts.DistanceWeight, ...
        'Standardize', classifierOpts.Standardize, ...
        'ClassNames', classNames,...
        'OptimizeHyperparameters', classifierOpts.optimizeHyperparameters,...
        'HyperparameterOptimizationOptions',...
        struct('Optimizer','bayesopt',...
            'Verbose', 1,...
            'UseParallel',true,...
            'MaxObjectiveEvaluations',100,...
            'CVPartition', cvPartition,...
            'Repartition', false,...
            'AcquisitionFunctionName','expected-improvement-plus'));
    fprintf('finished. ');
    toc
    fprintf('\n');

end

