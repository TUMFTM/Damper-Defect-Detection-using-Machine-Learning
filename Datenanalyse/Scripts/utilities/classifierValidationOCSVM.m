function result = classifierValidationOCSVM(predictionScore,labelAsMatrix,uniqueClasses,varargin)
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
    
    % set classifier
    if find(strcmp(varargin,'Classifier'))
        classifier = varargin{find(strcmp(varargin,'Classifier'))+1};
    else
        classifier = [];
    end

    if size(predictionScore,2) > size(predictionScore,1)
        predictionScore = predictionScore';
    end
    
    %% Some general variables
    result = struct();
    fprintf('\nStarting Classifier Validation...');


    %% Model validation
    numObs = length(predictionScore);

    % Get unique classes and sort as specified 
    classes = {};
    classes = vertcat(classes,uniqueClasses(contains(uniqueClasses,'Damper')));
    classes = vertcat(classes,uniqueClasses(contains(uniqueClasses,'Toe')));
    classes = vertcat(classes,uniqueClasses(contains(uniqueClasses,'Spring')));
    classes = vertcat(classes,uniqueClasses(~contains(uniqueClasses,classes)));
    
    if max(strcmp(classes,'good'))
        classDetection = 'good';
    elseif max(strcmp(classes,'passive intact'))
        classDetection = 'passive intact';
    elseif max(strcmp(classes,'passiveIntact'))
        classDetection = 'passiveIntact';
    elseif max(strcmp(classes,'allDampersStiff'))
        classDetection = 'allDampersStiff';
    end

    
    % Probability based predictions
    predictionScoreProbability = predictionScore;
    labelAsMatrixProbability = labelAsMatrix;
    IdxNoPrediction = max(predictionScore,[],2) < minPosteriorProbability;
    predictionScoreProbability(IdxNoPrediction,:) = [];
    labelAsMatrixProbability(IdxNoPrediction,:) = [];

    
    %% Calculate performance indicators
    % Multiclass
    [cMulticlass,cmMulticlass,ind,perMulticlass] = confusion(labelAsMatrix',predictionScore');
    cmMulticlass = cmMulticlass'; % Transpose indexing to make it fit with 'plotconfusion'-plot
    result.multiclass.raw.accuracy = 1-cMulticlass;
    result.multiclass.raw.falseOmissionRate = mean(perMulticlass(:,1));
    result.multiclass.raw.falseDiscoveryRate = mean(perMulticlass(:,2));
    result.multiclass.raw.precision = mean(perMulticlass(:,3));
    result.multiclass.raw.specificity = mean(perMulticlass(:,4));
    result.multiclass.raw.recall = mean(diag(cmMulticlass)'./sum(cmMulticlass,1));
    result.multiclass.raw.negativePredictiveValue = mean(diag(cmMulticlass)./sum(cmMulticlass,2));
    result.multiclass.raw.F1score = 2 * result.multiclass.raw.precision * result.multiclass.raw.recall / (result.multiclass.raw.precision + result.multiclass.raw.recall);
    
    % Multicalss probability based
    [cMulticlassProb,cmMulticlassProb,ind,perMulticlassProb] = confusion(labelAsMatrixProbability',predictionScoreProbability');
    cmMulticlassProb = cmMulticlassProb';
    result.multiclass.prob.accuracy = 1-cMulticlassProb;
    result.multiclass.prob.falseOmissionRate = mean(perMulticlassProb(:,1));
    result.multiclass.prob.falseDiscoveryRate = mean(perMulticlassProb(:,2));
    result.multiclass.prob.precision = mean(perMulticlassProb(:,3));
    result.multiclass.prob.specificity = mean(perMulticlassProb(:,4));
    result.multiclass.prob.recall = mean(diag(cmMulticlassProb)'./sum(cmMulticlassProb,1));
    result.multiclass.prob.negativePredictiveValue = mean(diag(cmMulticlassProb)./sum(cmMulticlassProb,2));
    result.multiclass.prob.F1score = 2 * result.multiclass.prob.precision * result.multiclass.prob.recall / (result.multiclass.prob.precision + result.multiclass.prob.recall);
    result.multiclass.prob.relReductionPredictions = 1-sum(sum(cmMulticlassProb))/(sum(sum(cmMulticlass)));

    %% Plot Confusion Matrix
    if ctrl.generateConfusionMatrix
        figure;
        plotconfusion(labelAsMatrix',predictionScore','Multiclass')
        title('Multiclass');
        set(gca,'XTickLabel',[classes;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classes;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
        figure;
        plotconfusion(labelAsMatrixProbability',predictionScoreProbability','Multiclass Probability-based');
        title('Multiclass Probability-based');
        set(gca,'XTickLabel',[classes;' ']);
        set(gca,'XTickLabelRotation',60);
        set(gca,'YTickLabel',[classes;' ']);
        set(findall(gcf,'-property','FontSize'),'FontSize',18)
        
    end

%     %% Quantify Isolation Performance 
%     % https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
%     for cntClass = 1 : size(classes,1)
%         currClass = classes{cntClass};
%         result.(currClass) = calcClassifierPerformance(trueClass,predictedClass,currClass,predictionScore,classifier,minPosteriorProbability);
%     end
%     % Calculate Performance for Detection
%     currClass = 'defect';
%     result.(currClass) = calcClassifierPerformance(labelAsMatrixDetection,predictionScoreDetection,currClass,predictionScore,classifier,minPosteriorProbability);
% 
%     %% Calculating overall performance by averaging all performance scores
%     fieldsAnalysis = fields(result.(classes{1}).raw);
%     for cntFields = 1 : size(fieldsAnalysis,1)
%         result.average.raw.(fieldsAnalysis{cntFields}) = 0;
%         result.average.prob.(fieldsAnalysis{cntFields}) = 0;
%         for cntClass = 1 : size(classes,1)
%             result.average.raw.(fieldsAnalysis{cntFields}) = result.average.raw.(fieldsAnalysis{cntFields}) + result.(classes{cntClass}).raw.(fieldsAnalysis{cntFields});
%             result.average.prob.(fieldsAnalysis{cntFields}) = result.average.prob.(fieldsAnalysis{cntFields}) + result.(classes{cntClass}).prob.(fieldsAnalysis{cntFields});
%         end
%         result.average.raw.(fieldsAnalysis{cntFields}) = result.average.raw.(fieldsAnalysis{cntFields}) / size(classes,1);
%         result.average.prob.(fieldsAnalysis{cntFields}) = result.average.prob.(fieldsAnalysis{cntFields}) / size(classes,1);
%     end


    %% Text Output
    fprintf('finished\n');
    
    % Multiclass
    fprintf('Average Multiclass: %.0f observations with %.0f classes - selected minimal probability of %.0f %% (%.2f %% predictions lost)\n', numObs, size(classes,1), minPosteriorProbability*100, result.multiclass.prob.relReductionPredictions*100);
    fprintf('  Accuracy (true predictions (positive and negative) / all predictions):              %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.accuracy, 100*result.multiclass.prob.accuracy);
    fprintf('  Recall (true positive predictions / real positive observations):                    %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.recall, 100*result.multiclass.prob.recall);
    fprintf('  Precision (true positive predictions / all positive predictions):                   %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.precision, 100*result.multiclass.prob.precision);
    fprintf('  F1-Score (harmonic mean of Precision and Recall):                                   %.2f %% (%.2f %% probability based) \n', 100*result.multiclass.raw.F1score, 100*result.multiclass.prob.F1score);
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

