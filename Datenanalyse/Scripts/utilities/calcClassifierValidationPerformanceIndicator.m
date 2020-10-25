function result = calcClassifierValidationPerformanceIndicator(positives, negatives, truePositives, trueNegatives, falsePositives, falseNegatives)
%CALCCLASSIFIERVALIDATIONPERFORMANCEINDICATOR Summary of this function goes here
%   Detailed explanation goes here
%   https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
        
        % Recall or True Positive Rate (eqv. with hit rate or sensitivity)
        result.recall = truePositives / positives;

        % Specificity or True Negative Rate
        result.specificity = trueNegatives / negatives;

        % False Positive Rate
        result.falsePositiveRate = falsePositives / negatives;
        
        % False Negative Rate
        result.falseNegativeRate = falseNegatives / positives;
        
        
        % Precision or Positive Predictive Value
        result.precision = truePositives / (truePositives + falsePositives);
        
        % Negative Predictive Value
        result.negativePredictiveValue = trueNegatives / (trueNegatives + falseNegatives);

        % False Discovery Rate
        result.falseDiscoveryRate = falsePositives / (truePositives + falsePositives);

        % False Omission Rate
        result.falseOmissionRate = falseNegatives / (trueNegatives + falseNegatives);

        
        % Accuracy
%         result.accuracy = (truePositives + trueNegatives) / ...
%             (truePositives + falsePositives + trueNegatives + falseNegatives);
        result.accuracy = (truePositives + trueNegatives) / ...
                    (positives + negatives);

        % Balanced Accuracy
        result.balancedAccuracy = (truePositives / positives + ...
            trueNegatives / negatives) / 2;

        % F1-Score
%         result.F1score = 2* truePositives / ...
%             (2 * truePositives + falsePositives + falseNegatives);
        result.F1score = 2 * (result.recall * result.precision) / ...
            (result.recall + result.precision);

        % Matthews Correlation Coefficient (MCC) (https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
        try
            result.MCC = (truePositives * trueNegatives - ...
                falsePositives * falseNegatives) / ...
                sqrt( (truePositives + falsePositives) * ...
                (truePositives + falseNegatives) * ...
                (trueNegatives + falsePositives) * ...
                (trueNegatives + falseNegatives));
        catch
            result.MCC = [];
        end
end

