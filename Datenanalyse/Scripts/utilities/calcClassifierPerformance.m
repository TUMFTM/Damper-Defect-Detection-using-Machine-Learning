function [output] = calcClassifierPerformance(trueClass,predictedClass,currClass,Posterior,classifier,minPosteriorProbability)
%CALCCLASSIFIERPERFORMANCE Summary of this function goes here
%   Detailed explanation goes here

        output = struct();

        % Positives - is class 'currClass'
        output.positives = sum(strcmp(trueClass,currClass));

        % Negatives - is not class 'currClass'
        output.negatives = sum(~strcmp(trueClass,currClass));

        % True Positives (TP) - eqv. with hit; correctly classified as positive
        truePositives = sum(strcmp(predictedClass,currClass) .* strcmp(trueClass,currClass));

        % True Negatives (TN) - eqv. with correct rejection; correctly classified as negative
        trueNegatives = sum((~strcmp(predictedClass,currClass)) .* (~strcmp(trueClass,currClass)));

        % False Positives (FP) - eqv. with false alarm, Type I error; falsely classified as positive
        falsePositives = sum(strcmp(predictedClass,currClass) .* (~strcmp(trueClass,currClass)));

        % False Negatives (FN) - eqv. with miss, Type II error; falsely classified as negative
        falseNegatives = sum((~strcmp(predictedClass,currClass)) .* strcmp(trueClass,currClass));

        % Calculate classifier validiation performance indicator
        output.raw = calcClassifierValidationPerformanceIndicator(output.positives, ...
            output.negatives, truePositives, ...
            trueNegatives, falsePositives, falseNegatives);

        % Copy data (needed because struct field would be deleted otherwise)
        output.raw.truePositives = truePositives;
        output.raw.trueNegatives = trueNegatives;
        output.raw.falsePositives = falsePositives;
        output.raw.falseNegatives = falseNegatives;
        
        %% Posterior Probability based performance
        currClassIdx = strcmp(currClass,classifier.ClassNames);
        % True Positives (TP) - eqv. with hit; correctly classified as positive
        truePositives = sum(strcmp(predictedClass,currClass) .* strcmp(trueClass,currClass) .* (Posterior(:,currClassIdx) > minPosteriorProbability));

        % True Negatives (TN) - eqv. with correct rejection; correctly classified as negative
        trueNegatives = sum((~strcmp(predictedClass,currClass)) .* (~strcmp(trueClass,currClass)) .* (Posterior(:,currClassIdx) < (1 - minPosteriorProbability)));

        % False Positives (FP) - eqv. with false alarm, Type I error; falsely classified as positive
        falsePositives = sum(strcmp(predictedClass,currClass) .* (~strcmp(trueClass,currClass)) .* (Posterior(:,currClassIdx) > minPosteriorProbability));

        % False Negatives (FN) - eqv. with miss, Type II error; falsely classified as negative
        falseNegatives = sum((~strcmp(predictedClass,currClass)) .* strcmp(trueClass,currClass) .* (Posterior(:,currClassIdx) < (1 - minPosteriorProbability)));
        
        % Calculate classifier validiation performance indicator
        output.prob = calcClassifierValidationPerformanceIndicator(output.positives, ...
            output.negatives, truePositives, ...
            trueNegatives, falsePositives, falseNegatives);

        % Copy data (needed because struct field would be deleted otherwise)
        output.prob.truePositives = truePositives;
        output.prob.trueNegatives = trueNegatives;
        output.prob.falsePositives = falsePositives;
        output.prob.falseNegatives = falseNegatives;

end

