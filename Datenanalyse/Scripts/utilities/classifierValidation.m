function result = classifierValidation(predictionScoreIn,labelAsMatrix,uniqueClasses,varargin)
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

    if ~iscell(predictionScoreIn)
        predictionScoreIn = {predictionScoreIn};
    end
    
    for cntCV = 1 : length(predictionScoreIn)
    
        predictionScore = predictionScoreIn{cntCV};
        
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
        elseif max(strcmp(classes,'intact'))
            classDetection = 'intact';
        end

        if exist('classDetection','var')
            % PredictionScoreDetection entspricht der
            % passiveIntact-Wahrscheinlichkeit verglichen mti der gr��ten
            % Defekt-Wahrscheinlichkeit
            predictionScoreDetection = zeros(length(labelAsMatrix),2);
            predictionScoreDetection(:,2) = predictionScore(:,strcmp(uniqueClasses,classDetection));
            if ~strcmp(uniqueClasses,classDetection)
                predictionScoreDetection(:,1) = max(predictionScore(:,~strcmp(uniqueClasses,classDetection)),[],2);
                predictionScoreDetection = predictionScoreDetection ./ sum(predictionScoreDetection,2);
            else
                predictionScoreDetection(:,1) = 1 - predictionScoreDetection(:,2);
            end
            labelAsMatrixDetection = zeros(length(labelAsMatrix),2);
            labelAsMatrixDetection(:,2) = labelAsMatrix(:,strcmp(uniqueClasses,classDetection));
            labelAsMatrixDetection(:,1) = 1 - labelAsMatrixDetection(:,2);

            classesDetection{1,1} = 'defect';
            classesDetection{2,1} = classDetection;
        end

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

        % Detection 
        if size(labelAsMatrix,2)>2
            [cDetection,cmDetection,ind,perDetection] = confusion(labelAsMatrixDetection',predictionScoreDetection');
            cmDetection = cmDetection';
            result.defect.raw.accuracy = 1-cDetection;
            result.defect.raw.falseOmissionRate = cmDetection(2,1)/(cmDetection(2,1)+cmDetection(2,2));
            result.defect.raw.falseDiscoveryRate = cmDetection(1,2)/(cmDetection(1,1)+cmDetection(1,2));
            result.defect.raw.precision = cmDetection(1,1)/(cmDetection(1,1)+cmDetection(1,2));
            result.defect.raw.specificity = cmDetection(2,2)/(cmDetection(1,2)+cmDetection(2,2));
            result.defect.raw.recall = cmDetection(1,1)/(cmDetection(1,1)+cmDetection(2,1));
            result.defect.raw.negativePredictiveValue = cmDetection(2,2)/(cmDetection(2,1)+cmDetection(2,2));
            result.defect.raw.F1score = 2 * result.defect.raw.precision * result.defect.raw.recall / (result.defect.raw.precision + result.defect.raw.recall);
            result.defect.positives = cmDetection(1,1) + cmDetection(2,1);
            result.defect.negatives = cmDetection(1,2) + cmDetection(2,2);
        end


        %% Calculate m-score (quasi multiclass AUC)
        result.multiclass.raw.mValue = multiClassAUC(predictionScore,vec2ind(labelAsMatrix')');
        if size(labelAsMatrix,2)>2
            result.defect.raw.mValue = multiClassAUC(predictionScoreDetection, vec2ind(labelAsMatrixDetection')');
        end

        %% Plot Confusion Matrix
        if ctrl.generateConfusionMatrix
            figure;
            plotconfusion(labelAsMatrix',predictionScore','Multiclass')
            title('Multiclass');
            set(gca,'XTickLabel',[classes;' ']);
            set(gca,'XTickLabelRotation',60);
            set(gca,'YTickLabel',[classes;' ']);
            set(findall(gcf,'-property','FontSize'),'FontSize',18)

            if size(labelAsMatrix,2)>2
                % Detection
                figure;
                plotconfusion(labelAsMatrixDetection',predictionScoreDetection','Detection');
                title('Detection');
                set(gca,'XTickLabel',[classesDetection;' ']);
                set(gca,'XTickLabelRotation',60);
                set(gca,'YTickLabel',[classesDetection;' ']);
                set(findall(gcf,'-property','FontSize'),'FontSize',18)
            end

        end

        if cntCV == 1
            resultOut = result;
        else
            resultOut = mergeStructs(resultOut, result, 1);
        end
    end
    
    clear result
    fields1 = fields(resultOut);
    for cnt1 = 1 : length(fields1)
        fields2 = fields(resultOut.(fields1{cnt1}));
        for cnt2 = 1 : length(fields2)
            try
                result.mean.(fields1{cnt1}).(fields2{cnt2}) = structfun(@mean,resultOut.(fields1{cnt1}).(fields2{cnt2}),'UniformOutput',false);
                result.std.(fields1{cnt1}).(fields2{cnt2}) = structfun(@std,resultOut.(fields1{cnt1}).(fields2{cnt2}),'UniformOutput',false);
            end
        end
    end
    
    %% Text Output
    fprintf('finished\n');
    
    % Multiclass
    fprintf('Average Multiclass: %.0f observations with %.0f classes\n', numObs, size(classes,1));
    fprintf('  Accuracy (true predictions (positive and negative) / all predictions):              %.2f +- %.2f %%\n', 100*result.mean.multiclass.raw.accuracy, 100*result.std.multiclass.raw.accuracy);
    fprintf('  Recall (true positive predictions / real positive observations):                    %.2f +- %.2f %%\n', 100*result.mean.multiclass.raw.recall, 100*result.std.multiclass.raw.recall);
    fprintf('  Precision (true positive predictions / all positive predictions):                   %.2f +- %.2f %%\n', 100*result.mean.multiclass.raw.precision, 100*result.std.multiclass.raw.precision);
    fprintf('  F1-Score (harmonic mean of Precision and Recall):                                   %.2f +- %.2f %%\n', 100*result.mean.multiclass.raw.F1score, 100*result.std.multiclass.raw.F1score);
    fprintf('  m-value:                                                                            %.2f +- %.2f %%\n', 100*result.mean.multiclass.raw.mValue, 100*result.std.multiclass.raw.mValue);

    % Detection
    if size(labelAsMatrix,2)>2
        fprintf('-------------\n');
        fprintf('Used Class for Detection: %s = negative, all other classes = positive\n', classDetection);
        % fprintf('Detection: %.0f real positive observations, %.0f real negative observations\n', result.mean.defect.positives, result.mean.defect.negatives);
        fprintf('  Accuracy (true predictions (positive and negative) / all predictions):         %.2f +- %.2f %%\n', 100*result.mean.defect.raw.accuracy, 100*result.std.defect.raw.accuracy);
        fprintf('  Recall (true positive predictions / real positive observations):               %.2f +- %.2f %%\n', 100*result.mean.defect.raw.recall, 100*result.std.defect.raw.recall);
        fprintf('  Precision (true positive predictions / all positive predictions):              %.2f +- %.2f %%\n', 100*result.mean.defect.raw.precision, 100*result.std.defect.raw.precision);
        fprintf('  F1-Score (harmonic mean of Precision and Recall):                              %.2f +- %.2f %%\n', 100*result.mean.defect.raw.F1score, 100*result.std.defect.raw.F1score);
        fprintf('  m-value:                                                                       %.2f +- %.2f %%\n', 100*result.mean.defect.raw.mValue, 100*result.std.defect.raw.mValue);
    end
    
    fprintf('-------------\n');
    fprintf('Probability based numbers sind relativ zur Anzahl der reduzierten abgegebenen Klassifikationen\n');

end

