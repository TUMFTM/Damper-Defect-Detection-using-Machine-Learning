function [posteriorProbabilityMatrix, posteriorPredictedClassSequence] = calcSeriesProbabilityForEachMeasurement(featuresTesting, featuresTraining, trainedClassifier)
%CALCULATEMEANPROBABILITY Summary of this function goes here
%   Calculation of series probability according to
%   S. Lenser and M. Veloso, “Non-Parametric Time Series Classification,” 
%   in 2005 IEEE International Conference on Robotics and Automation (ICRA): 
%   Barcelona, Spain : 18-22 April, 2005, Barcelona, Spain, 2005, pp. 3918–3923.

    posteriorProbabilityMatrix = zeros(size(featuresTesting.Prop.Posterior));
    distance = 'euclidean';     % use 'euclidean' (default) | 'seuclidean' | 'cityblock' | 'chebychev' | 'minkowski' | 'mahalanobis'
    
    alpha = 0.95;
    transitionMatrix = alpha * eye(size(featuresTesting.Prop.Posterior,2)) + (1-alpha) / (size(featuresTesting.Prop.Posterior,2)-1) * (ones(size(featuresTesting.Prop.Posterior,2))-eye(size(featuresTesting.Prop.Posterior,2)));
    
    % Testweise wurde die confusionmatrix als transitionMatrix genommen.
    % Kann man machen, führt einfach zu "anderen" Ergebnissen
%     transitionMatrix = confusionMatrix;
%     if size(transitionMatrix,1)>size(features.Prop.Posterior,2)
%         transitionMatrix = transitionMatrix(2:end,:)';
%     end
    
%     if istable(featuresTraining.data)
%         featuresTraining.data = table2array(featuresTraining.data);
%     end
%     if istable(featuresTesting.data)
%         featuresTesting.data = table2array(featuresTesting.data);
%     end

    % Prediction für jeweils nur die relevanten Punkte aus den
    % Trainingsdaten ist für Serienimplementierung geschickter.
    % Für Entwicklung aber auskommentiert.
    if isempty(trainedClassifier)
        aprioriProbabilityMatrix = featuresTraining.Prop.Posterior;
    else
        [~, aprioriProbabilityMatrix] = predictClassifier(trainedClassifier, featuresTraining.data);
    end
    
    idxOP = [false; diff(featuresTraining.Prop.batch)==0];
    OP.base_values = aprioriProbabilityMatrix(find(idxOP)-1,:);
    OP.output_value = aprioriProbabilityMatrix(idxOP,:);
    
    batch_vec = unique(featuresTesting.Prop.batch);
    
    for idxBatch = 1 : length(batch_vec)
        
        cntBatch = batch_vec(idxBatch);
        
        % Identify relevant observations of current measurement
        idxMeasurement = logical(featuresTesting.Prop.batch == cntBatch);
        idxMeasurementTrue = find(idxMeasurement);
        
        sqrt_n = floor(sqrt(sum(idxMeasurement)));
        base_values = featuresTesting.Prop.Posterior(idxMeasurement,:);
        [idxKNN,distKNN] = knnsearch(OP.base_values,base_values,'K',sqrt_n,'Distance',distance);
        h_b = distKNN(:,end);
        
        %% Calculate posterior Probability of the current measurement
        for cntIdxMeasTrue = 1 : length(idxMeasurementTrue)

            cntIdx = idxMeasurementTrue(cntIdxMeasTrue);

            if cntIdxMeasTrue == 1
                % Use first probability as first estimate
%                 posteriorProbabilityMatrix(cntIdx,:) = featuresTesting.Prop.Posterior(cntIdx,:);
                
                % Use mean probability of measurement as first estimate
                posteriorProbabilityMatrix(cntIdx,:) = mean(featuresTesting.Prop.Posterior(idxMeasurement,:),1);
            else
%                 sqrt_n = floor(sqrt(sum(idxMeasurement)));
%                 base_values = featuresTesting.Prop.Posterior(cntIdx,:);
%                 
%                 [idxKNN,distKNN] = knnsearch(aprioriProbabilityMatrix,base_values,'K',sqrt_n,'Distance',distance);
%                 
%                 h_b = distKNN(:,end);
                
                % Prediction für jeweils nur die relevanten Punkte aus den
                % Trainingsdaten ist für Serienimplementierung geschickter.
                % Für Entwicklung aber auskommentiert.
%                 [~, aprioriProbabilityMatrix] = predictClassifier(trainedClassifier, featuresTraining.data(idxKNN,:));
                pred = OP.output_value(idxKNN(cntIdxMeasTrue,:),:);
                
                % Perform correlation correction on pred (not implemented
                % yet)
%                 base = featuresTraining.data(idxKNN,:);
                base = OP.base_values(idxKNN(cntIdxMeasTrue,:),:);

                z = 0 : 0.01 : 1;
                pdf = zeros(length(z), size(featuresTesting.Prop.Posterior,2));
                for cntClass = 1 : size(featuresTesting.Prop.Posterior,2)
                    tmppdf = zeros(size(base,1),length(z));
                    for i = 1 : size(base,1)
                        h_io = 0.5 * abs(featuresTesting.Prop.Posterior(cntIdx-1,cntClass) - pred(i,cntClass));     % Calculate h_io distance
                        tmppdf(i,:) = K_g(pred(i,cntClass)*ones(size(z))-z, h_io) * K_t(base(i,:)-base_values(cntIdxMeasTrue,:), h_b(cntIdxMeasTrue,:), distance);
                    end
                    pdf(:,cntClass) = sum(tmppdf,1)';
                end
                
                Probability_xj = zeros(1,size(featuresTesting.Prop.Posterior,2));
                for cntClass = 1 : size(featuresTesting.Prop.Posterior,2)
                    Probability_xj(1,cntClass) = normfit(z',0.05,zeros(size(z')),pdf(:,cntClass));
                end
%                 Probability_xj = Probability_xj ./ sum(Probability_xj);
                
                % neue Berechnung mit Probable Series Predictor
                posteriorProbabilityMatrix(cntIdx,:) = Probability_xj .* posteriorProbabilityMatrix(cntIdx-1,:) * transitionMatrix;
                posteriorProbabilityMatrix(cntIdx,:) = posteriorProbabilityMatrix(cntIdx,:) / sum(posteriorProbabilityMatrix(cntIdx,:));
                
                % Wenn die Anzahl der Beobachtungen einer Messung zu klein
                % ist (3 oder weniger Beobachtungen) werden NaNs generiert.
                % Diese werden hier abgefangen. Die Wahrscheinlichkeit
                % entspricht dann der aPriori Wahrscheinlichkeit.
                if isnan(posteriorProbabilityMatrix(cntIdx,:))
                    posteriorProbabilityMatrix(cntIdx,:) = featuresTesting.Prop.Posterior(cntIdx,:);
                end

            end
        end

    end

    % Generate Sequence of posterior predicted classes
    [~,posteriorPredictedClassSequence] = max(posteriorProbabilityMatrix,[],2);
    
end

function K_t = K_t(x,h,distance)
    
    if strcmp(distance,'euclidean')
        normxh = norm(x)/h;
        if normxh <= 1
            K_t = (1-(normxh)^2)^3;
        else
            K_t = 0;
        end
    end
end

function K_g = K_g(x,h)
    
    if length(x)>1
        K_g = zeros(size(x));
        idx = logical(abs(x./h)<=1);
        K_g(idx) = 0.9999;
        K_g(~idx) = 0.0001;
    else
        normxh = norm(x)/h;
        if normxh <= 1
            K_g = 0.9999;
        else
            K_g = 0.0001;
        end
    end
end