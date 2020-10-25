function [trainedClassifier] = trainSVM(datastructTraining, classifierOpts, varargin)
% [trainedClassifier] = trainClassifier(trainingData, SVMopts)
% returns a trained classifier and its accuracy. 
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


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.

    if find(strcmp(varargin,'TextOutput'))
        TextOutput = varargin{find(strcmp(varargin,'TextOutput'))+1};
    else
        TextOutput = 1;
    end
    
    if find(strcmp(varargin,'FitPosterior'))
        fitPosterior = varargin{find(strcmp(varargin,'FitPosterior'))+1};
    else
        fitPosterior = true;
    end

    % Convert input data to desired table format if it is a struct
    if isstruct(datastructTraining)
        response = datastructTraining.Prop.labelIsolation;
%         response = datastructTraining.Label;
%         dataTraining = datastructTraining.data;
    else
%         try
%             response = datastructTraining.Label;
%         catch
%             response = datastructTraining.Label_ISOLATION;
%         end
    end
    
    if find(strcmp(varargin,'selectedFeatures'))
        predictorNames = varargin{find(strcmp(varargin,'selectedFeatures'))+1};
    else
        predictorNames = datastructTraining.data.Properties.VariableNames';
    end

    if ~isfield(classifierOpts, 'ClassNames')
        classNames = unique(response);
    else
        classNames = classifierOpts.ClassNames;
    end

    % Create Partition for Holdout Validation
    if classifierOpts.validation <= 1
        % Ensure that fraction of validation dataset is correct (validation dataset < 50% of training data)
        classifierOpts.validation = min(classifierOpts.validation, 1-classifierOpts.validation);
        cvPartition = cvpartition(response,'HoldOut',classifierOpts.validation);
    else
        cvPartition = cvpartition(response,'kFold',classifierOpts.validation);
    end

    % Train a classifier
    if TextOutput
        fprintf('Classifier training...');
        tic
    end
    if ~strcmp(classifierOpts.optimizeHyperparameters,'')
        % Perform hyperparameter optimization
        optimStruct = struct('Optimizer','bayesopt',...
            'Verbose', 1,...
            'UseParallel',true,...
            'MaxObjectiveEvaluations',classifierOpts.MaxObjectiveEvaluations,...
            'CVPartition', cvPartition,...
            'Repartition', false,...
            'AcquisitionFunctionName','expected-improvement-plus');
        classificationSVM = fitcecoc(...
            datastructTraining.data, ...
            response, ...
            'PredictorNames', predictorNames, ...
            'Learners', classifierOpts.template, ...
            'Coding', classifierOpts.Coding, ...
            'FitPosterior', fitPosterior,... % 'CVPartition', cvPartition,...
            'ClassNames', classNames,...
            'Options',statset('UseParallel',true),...
            'OptimizeHyperparameters', classifierOpts.optimizeHyperparameters,...
            'HyperparameterOptimizationOptions',optimStruct);
        if classifierOpts.validation > 1
            classificationSVM = crossval(classificationSVM, 'KFold', classifierOpts.validation);
        end

    else

        % Train classifier
        classificationSVM = fitcecoc(...
            datastructTraining.data, ...
            response, ...
            'PredictorNames', predictorNames, ...
            'Learners', classifierOpts.template, ...
            'Coding', classifierOpts.Coding, ...
            'FitPosterior', fitPosterior,...
            'CVPartition', cvPartition,...
            'ClassNames', classNames,...
            'Options',statset('UseParallel',true));

    end

    if isprop(classificationSVM, 'Trained')
        if length(classificationSVM.Trained)==1
            trainedClassifier = classificationSVM.Trained{1,1};
        else
            trainedClassifier = classificationSVM.Trained;
        end
    else
        trainedClassifier = classificationSVM;
    end

    if TextOutput
        fprintf('finished. ');
        toc
    end

end


