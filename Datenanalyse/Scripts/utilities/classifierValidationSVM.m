function [result, predictedClass] = classifierValidationSVM(classifier,featureTable,varargin)
%classifierValidation Validation of classification model 
%   Detailed explanation goes here

    % set control variable for confusion matrix generation
    if find(strcmp(varargin,'generateConfusionMatrix'))
        ctrl.generateConfusionMatrix = varargin{find(strcmp(varargin,'generateConfusionMatrix'))+1};
    else
        ctrl.generateConfusionMatrix = 0;
    end
    
    % set confidence level
    if find(strcmp(varargin,'minPosteriorProbability'))
        minPosteriorProbability = varargin{find(strcmp(varargin,'minPosteriorProbability'))+1};
    else
        minPosteriorProbability = 0.4;
    end

    %% Some general variables
    result = struct();
    fprintf('\nStarting Classifier Validation...');


    %% Model validation
    [numObs,~] = size(featureTable);
    trueClass = featureTable.Label;

    % Get unique classes and sort as specified 
    tmpClasses = unique(featureTable.Label);
    classes = {};
    classes = vertcat(classes,tmpClasses(contains(tmpClasses,'Damper')));
    classes = vertcat(classes,tmpClasses(contains(tmpClasses,'Toe')));
    classes = vertcat(classes,tmpClasses(contains(tmpClasses,'Spring')));
    classes = vertcat(classes,tmpClasses(~contains(tmpClasses,classes)));
    
    if max(strcmp(classes,'good'))
        classDetection = 'good';
    elseif max(strcmp(classes,'passive intact'))
        classDetection = 'passive intact';
    elseif max(strcmp(classes,'passiveIntact'))
        classDetection = 'passiveIntact';
    end
    
    % Predict class
    try
    [predictedClass, NegLoss, PBScore, Posterior] = predict(classifier, table2array(featureTable(:,1:end-1)));
    catch
        predictedClass = classifier.predictFcn(featureTable(:,1:end-1));
        Posterior = zeros(size(featureTable,1),size(classes,1));
        NegLoss = zeros(size(featureTable,1),size(classes,1));
    end
    
        
    if exist('classDetection','var')
        predictedClassDetection = predictedClass;
        predictedClassDetection(~strcmp(predictedClass,classDetection)) = {'defect'};
        
        trueClassDetection = trueClass;
        trueClassDetection(~strcmp(trueClassDetection,classDetection)) = {'defect'};
    end
    
    
    
    %% Create Confusion Matrix
    % Create matrix with class indexes for confusion matrix generation
    IdxTrueClass = zeros(size(NegLoss));
    IdxPredictedClass = zeros(size(NegLoss));
    for cntClasses = 1 : size(classes,1)
        IdxTrueClass(:,cntClasses) = strcmp(trueClass,classes{cntClasses});
        IdxPredictedClass(:,cntClasses) = strcmp(predictedClass,classes{cntClasses});
    end

    classesDetection = unique(trueClassDetection);

    IdxTrueClassDetection = zeros(size(NegLoss,1),size(classesDetection,1));
    IdxPredictedClassDetection = zeros(size(NegLoss,1),size(classesDetection,1));

    IdxTrueClassDetection(:,1) = strcmp(trueClassDetection,classesDetection{1});
    IdxTrueClassDetection(:,2) = strcmp(trueClassDetection,classesDetection{2});
    IdxPredictedClassDetection(:,1) = strcmp(predictedClassDetection,classesDetection{1});
    IdxPredictedClassDetection(:,2) = strcmp(predictedClassDetection,classesDetection{2});
    
%     % invert class labels for detection for having "defect" as positive class
%     IdxTrueClassDetection = 1 - IdxTrueClassDetection;
%     IdxPredictedClassDetection = 1 - IdxPredictedClassDetection;
    
    % Probability based predictions
    IdxPredictedClassProbability = IdxPredictedClass;
    IdxTrueClassProbability = IdxTrueClass;
    IdxNoPrediction = max(Posterior,[],2) < minPosteriorProbability;
    IdxPredictedClassProbability(IdxNoPrediction,:) = [];
    IdxTrueClassProbability(IdxNoPrediction,:) = [];
    
    % Probability based predictions for Detection
    IdxPredictedClassDetectionProbability = IdxPredictedClassDetection;
    IdxTrueClassDetectionProbability = IdxTrueClassDetection;
    IdxNoPredictionDetection = max(Posterior,[],2) < minPosteriorProbability;   % entspricht: "nur dann eine Schätzung abgeben, wenn irgendeine Klasse auch über der gewählten Wahrscheinlichkeit liegt
    IdxPredictedClassDetectionProbability(IdxNoPredictionDetection,:) = [];
    IdxTrueClassDetectionProbability(IdxNoPredictionDetection,:) = [];
    
    if ctrl.generateConfusionMatrix
        figure;
        plotconfusion(IdxTrueClass',IdxPredictedClass','Multiclass')
        title('Multiclass');
        set(gca,'XTickLabel',[classes;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classes;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
        figure;
        plotconfusion(IdxTrueClassProbability',IdxPredictedClassProbability','Multiclass Probability-based');
        title('Multiclass Probability-based');
        set(gca,'XTickLabel',[classes;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classes;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
        
        % Detection
        figure;
        plotconfusion(IdxTrueClassDetection',IdxPredictedClassDetection','Detection');
        title('Detection');
        set(gca,'XTickLabel',[classesDetection;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classesDetection;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
        figure;
        plotconfusion(IdxTrueClassDetectionProbability',IdxPredictedClassDetectionProbability','Detection Probability-based');
        title('Detection Probability-based');
        set(gca,'XTickLabel',[classesDetection;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classesDetection;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
    end

    %% Quantify Isolation Performance 
    % https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    for cntClass = 1 : size(classes,1)
        currClass = classes{cntClass};
        result.(currClass) = calcClassifierPerformance(trueClass,predictedClass,currClass,Posterior,classifier,minPosteriorProbability);
    end
    % Calculate Performance for Detection
    currClass = 'defect';
    result.(currClass) = calcClassifierPerformance(trueClassDetection,predictedClassDetection,currClass,Posterior,classifier,minPosteriorProbability);

    %% Calculating overall performance by averaging all performance scores
    fieldsAnalysis = fields(result.(classes{1}).raw);
    for cntFields = 1 : size(fieldsAnalysis,1)
        result.average.raw.(fieldsAnalysis{cntFields}) = 0;
        result.average.prob.(fieldsAnalysis{cntFields}) = 0;
        for cntClass = 1 : size(classes,1)
            result.average.raw.(fieldsAnalysis{cntFields}) = result.average.raw.(fieldsAnalysis{cntFields}) + result.(classes{cntClass}).raw.(fieldsAnalysis{cntFields});
            result.average.prob.(fieldsAnalysis{cntFields}) = result.average.prob.(fieldsAnalysis{cntFields}) + result.(classes{cntClass}).prob.(fieldsAnalysis{cntFields});
        end
        result.average.raw.(fieldsAnalysis{cntFields}) = result.average.raw.(fieldsAnalysis{cntFields}) / size(classes,1);
        result.average.prob.(fieldsAnalysis{cntFields}) = result.average.prob.(fieldsAnalysis{cntFields}) / size(classes,1);
    end

    %% Calculate measure for multiclass case
    
    % Multiclass
    [cMulticlass,cmMulticlass,ind,per] = confusion(IdxTrueClass',IdxPredictedClass');
    % Transpose indexing to make it fit with 'plotconfusion'-plot
    cmMulticlass = cmMulticlass';
    
    result.multiclass.raw.accuracy = 1-cMulticlass;
    result.multiclass.raw.falseOmissionRate = mean(per(:,1));
    result.multiclass.raw.falseDiscoveryRate = mean(per(:,2));
    result.multiclass.raw.precision = mean(per(:,3));
    result.multiclass.raw.specificity = mean(per(:,4));
    result.multiclass.raw.recall = mean(diag(cmMulticlass)'./sum(cmMulticlass,1));
    result.multiclass.raw.negativePredictiveValue = mean(diag(cmMulticlass)./sum(cmMulticlass,2));
    result.multiclass.raw.F1score = 2 * result.multiclass.raw.precision * result.multiclass.raw.recall / (result.multiclass.raw.precision + result.multiclass.raw.recall);
    
    % Multicalss probability based
    [cMulticlassProb,cmMulticlassProb,ind,per] = confusion(IdxTrueClassProbability',IdxPredictedClassProbability');
    cmMulticlassProb = cmMulticlassProb';
    
    result.multiclass.prob.accuracy = 1-cMulticlassProb;
    result.multiclass.prob.falseOmissionRate = mean(per(:,1));
    result.multiclass.prob.falseDiscoveryRate = mean(per(:,2));
    result.multiclass.prob.precision = mean(per(:,3));
    result.multiclass.prob.specificity = mean(per(:,4));
    result.multiclass.prob.recall = mean(diag(cmMulticlassProb)'./sum(cmMulticlassProb,1));
    result.multiclass.prob.negativePredictiveValue = mean(diag(cmMulticlassProb)./sum(cmMulticlassProb,2));
    result.multiclass.prob.F1score = 2 * result.multiclass.prob.precision * result.multiclass.prob.recall / (result.multiclass.prob.precision + result.multiclass.prob.recall);
    result.multiclass.prob.relReductionPredictions = 1-sum(sum(cmMulticlassProb))/(sum(sum(cmMulticlass)));
    
    % Detection probability based
    [c,cm,ind,per] = confusion(IdxTrueClassDetectionProbability',IdxPredictedClassDetectionProbability');
    cm = cm';
    
    result.defect.prob.accuracy = 1-c;
    result.defect.prob.falseOmissionRate = cm(2,1)/(cm(2,1)+cm(2,2));
    result.defect.prob.falseDiscoveryRate = cm(1,2)/(cm(1,1)+cm(1,2));
    result.defect.prob.precision = cm(1,1)/(cm(1,1)+cm(1,2));
    result.defect.prob.specificity = cm(2,2)/(cm(1,2)+cm(2,2));
    result.defect.prob.recall = cm(1,1)/(cm(1,1)+cm(2,1));
    result.defect.prob.negativePredictiveValue = cm(2,2)/(cm(2,1)+cm(2,2));
    result.defect.prob.F1score = 2 * result.defect.prob.precision * result.defect.prob.recall / (result.defect.prob.precision + result.defect.prob.recall);
    result.defect.prob.relReductionPredictions = result.multiclass.prob.relReductionPredictions;

    %% Text Output
    fprintf('finished\n');
    
    % Average Multiclass
    fprintf('Average Multiclass: %.0f observations with %.0f classes - selected minimal probability of %.0f %% (%.2f %% predictions lost)\n', numObs, size(classes,1), minPosteriorProbability*100, result.multiclass.prob.relReductionPredictions*100);
    fprintf('  Accuracy (true predictions (positive and negative) / all predictions):              %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.accuracy, 100*result.multiclass.prob.accuracy);
    fprintf('  Recall (true positive predictions / real positive observations):                    %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.recall, 100*result.multiclass.prob.recall);
    fprintf('  Precision (true positive predictions / all positive predictions):                   %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.precision, 100*result.multiclass.prob.precision);
    fprintf('  F1-Score (harmonic mean of Precision and Recall):                                   %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.F1score, 100*result.multiclass.prob.F1score);
%     fprintf('  Specificity (true negative predictions / real negative observations):               %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.specificity, 100*result.average.prob.specificity);
%     fprintf('  Negative Predictive Value (true negative predictions / all negative predictions):   %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.negativePredictiveValue, 100*result.average.prob.negativePredictiveValue);
%     fprintf('Average Multiclass False Positive Rate (false positive predictions / real negative observations): %.2f %% (%.2f %% probability based) \n', 100*result.average.raw.falsePositiveRate, 100*result.average.prob.falsePositiveRate);
%     fprintf('Average Multiclass F1 Score (harmonic mean of Precision and Recall): %.2f %% (%.2f %% probability based) \n', 100*result.average.raw.F1score, 100*result.average.prob.F1score);

    % Detection
    if exist('classDetection','var')
        fprintf('-------------\n');
        fprintf('Used Class for Detection: %s = negative, all other classes = positive\n', classDetection);
        fprintf('Detection: %.0f real positive observations, %.0f real negative observations - selected minimal probability of %.0f %% (%.2f %% predictions lost)\n', result.defect.positives, result.defect.negatives, minPosteriorProbability*100, result.defect.prob.relReductionPredictions*100);
        fprintf('  Accuracy (true predictions (positive and negative) / all predictions):         %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.accuracy, 100*result.defect.prob.accuracy);
        fprintf('  Recall (true positive predictions / real positive observations):               %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.recall, 100*result.defect.prob.recall);
        fprintf('  Precision (true positive predictions / all positive predictions):              %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.precision, 100*result.defect.prob.precision);
        fprintf('  F1-Score (harmonic mean of Precision and Recall):                              %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.F1score, 100*result.defect.prob.F1score);
        
%         fprintf('  False Negative Rate (false negative predictions / real negative observations): %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.falseNegativeRate, 100*result.defect.prob.falseNegativeRate);
%         fprintf('  False Omission Rate (false negative predictions / all negative predictions):   %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.falseOmissionRate, 100*result.defect.prob.falseOmissionRate);
        
%         fprintf('  Specificity (true negative predictions / real negative observations): %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.specificity, 100*result.defect.prob.specificity);
%         fprintf('Detection F1 Score (harmonic mean of Precision and Recall): %.2f %% (%.2f %% probability based) \n', 100*result.defect.raw.F1score, 100*result.defect.prob.F1score);
    end
    
    fprintf('-------------\n');
    fprintf('Probability based numbers sind relativ zur Anzahl der reduzierten abgegebenen Klassifikationen\n');

%     %% Generate ROC for detection
%     if exist('classDetection','var')
%         currClassIdx = strcmp(classDetection,classifier.ClassNames);
%         tmpTrueClass = trueClass;
%         tmpTrueClass(find(~strcmp(trueClass,classDetection))) = {'defect'};
%         XCrit = 'fpr';
%         YCrit = 'npv';
% %         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx),classDetection,'XCrit','fnr','YCrit','ppv');
%         figure;
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx)./(sum(PBScore(:,~currClassIdx),2)),classDetection,'XCrit',XCrit,'YCrit',YCrit);
%         plot(X,Y,'DisplayName',[classDetection,'/sum(not-',classDetection,')'])
%         hold on
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx)./(max(PBScore(:,~currClassIdx),[],2)),classDetection,'XCrit',XCrit,'YCrit',YCrit);
%         plot(X,Y,'DisplayName',[classDetection,'/max(not-',classDetection,')'])
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx),classDetection);
%         plot(X,Y,'DisplayName',[classDetection,'-score'])
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx)-max(PBScore(:,~currClassIdx),[],2),classDetection,'XCrit',XCrit,'YCrit',YCrit);
%         plot(X,Y,'DisplayName',[classDetection,' - max(not-',classDetection,')'])
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(trueClass,PBScore(:,currClassIdx)-sum(PBScore(:,~currClassIdx),2),classDetection,'XCrit',XCrit,'YCrit',YCrit);
%         plot(X,Y,'DisplayName',[classDetection,' - sum(not-',classDetection,')'])
%         [X,Y,T,AUC,OPTROCPT] = perfcurve(tmpTrueClass,(sum(PBScore(:,~currClassIdx),2))./PBScore(:,currClassIdx),'defect');
%         plot(X,Y,'DisplayName',['sum(not-',classDetection,')/',classDetection])
%         xlabel('False Positive Rate - Anteil unerkannte Fehler / alle erkannte Fehler') 
%         ylabel(sprintf('Negative Predictive Value \nAnteil richtig erkannte Fehler / alle erkannte Fehler'))
%         legend show
%         title('ROC for Failure Detection');
%     end

end

