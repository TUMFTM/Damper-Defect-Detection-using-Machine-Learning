function [predictedClass, Score] = predictClassifier(classifier, featuresAsTable)
%PREDICTCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

%     if ~isequal(classifier.ExpandedPredictorNames, featuresAsTable.Properties.VariableNames)
%         fprintf('Supplied features in %s not consistent to features of supplied classifier for prediction!!\n', mfilename);
%         return;
%     end
    classifierClass = class(classifier);
    switch classifierClass
        case 'cell' % CrossVal-mdl
            
            if nargout == 2
                predictedClass = cell(size(classifier));
                Score = cell(size(classifier));

                parfor cntCV = 1 : length(classifier)
                    fprintf('Starting Prediction %d of %d\n', cntCV, length(classifier));
                    tic;
                    [predictedClass{cntCV}, ~, ~, Score{cntCV}] = predict(classifier{cntCV}, featuresAsTable);
                    toc;
                end
                
            else
                predictedClass = cell(size(classifier));
                parfor cntCV = 1 : length(classifier)
                    fprintf('Starting Prediction %d of %d\n', cntCV, length(classifier));
                    tic;
                    [predictedClass{cntCV}] = predict(classifier{cntCV}, featuresAsTable);
                    toc;
                end
            end
            
%         case 'classreg.learning.classif.CompactClassificationECOC' % CrossVal-mdl
%             
%             predictedClass = cell(size(classifier.Trained));
%             Score = cell(size(classifier.Trained));
%             
%             for cntCV = 1 : length(classifier.Trained)
%                 [predictedClass{cntCV}, Score{cntCV}] = predict(classifier.Trained{cntCV}, featuresAsTable);
%             end
        
        case 'classreg.learning.classif.ClassificationEnsemble' % Decision Tree
            [predictedClass, Score] = predict(classifier, featuresAsTable);
            
            % Selfmade equation for adaption of scores:
            % maximal score is outputed if all weak learners vote for the
            % same class. This would result in a score =
            % classifier.TrainedWeights
            % random guessing is a score of 0 -> Proability score of
            % 1/numClasses
            Score = 1/length(classifier.ClassNames) + 1/length(classifier.ClassNames) * Score./sum(classifier.TrainedWeights);
        otherwise
            try
                % Classifier is SVM
                [predictedClass, ~, ~, Score] = predict(classifier, featuresAsTable);
            catch
                try
                    [predictedClass,Score,~] = predict(classifier, featuresAsTable);
                catch
                    fprintf('Prediction not possible - unkown classifier type\n');
                end
            end
            
    end
    
end

