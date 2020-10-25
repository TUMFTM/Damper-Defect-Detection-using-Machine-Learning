function features = featureExtraction(data,opts,varargin)
% Extracts features out of structured and cleaned-up raw data.
% Generates several features out of raw data.

    %% Specify Features to be calculated
    % set control variable for calculation of all features
    if find(strcmp(varargin,'calcAllFeatures'))
        ctrl.calcAllFeatures = varargin{find(strcmp(varargin,'calcAllFeatures'))+1};
    else
        ctrl.calcAllFeatures = 0;
    end

    % Detrend data
    data = detrendData(data, opts);
    
    if isfield(data,'Prop')
        dataProp = data.Prop;
        data = rmfield(data,'Prop');
    end

    %% "reduced" feature set, specified in master thesis of Alex Merk
    % ctrl.calcFeatures = {choose a unique name of the feature, function handle to calculate feature; ... (etc.)
    calcFeatures = {'FuncStdv', @()FuncStdv(data,opts);...
        'FuncARfit', @()FuncARfit(data,10,opts);...
        'FuncAbsEnergyTime', @()FuncAbsEnergyTime(data,opts);...
        'FuncSkewness', @()FuncSkewness(data,opts);...
        'FuncDiscrPower', @()FuncDiscrPower(data,5,opts);...
        'FuncEigNorm', @()FuncEigNorm(data,[10 15],[2.5 5],[4.3 5 3.5 3.5 3.5 3.5],[15 13.5 15 15 15 15],opts);...
        'FuncAbsSumChanges', @()FuncAbsSumChanges(data,opts);...
        'FuncCorCoeff', @()FuncCorCoeff(data,opts);...
        'FuncCrest', @()FuncCrest(data,opts);...
        'FuncNorm', @()FuncNorm(data,opts);...
        'FuncRange', @()FuncRange(data,opts);...
        'FuncComplTimeSeries', @()FuncComplTimeSeries(data,14,opts);...
        'FuncSVD', @()FuncSVD(data,opts);...
        'FuncSampEntr', @()FuncSampEntr(data,2,0.25,opts);...
        'FuncLogEnEntr', @()FuncLogEnEntr(data,opts);...
        'FuncMeanFreq', @()FuncMeanFreq(data,opts);...
        'FuncBandpower_1_5', @()FuncBandpower(data,1,5,opts);...
        'FuncBandpower_5_10', @()FuncBandpower(data,5,10,opts);...
        'FuncBandpower_10_15', @()FuncBandpower(data,10,15,opts);...
        'FuncBandpower_15_20', @()FuncBandpower(data,15,20,opts);...
        'FuncBandpower_20_25', @()FuncBandpower(data,20,25,opts)};


    %% calculate all features (reduced features with more features added)
    if ctrl.calcAllFeatures
        calcFeatures = [calcFeatures;...
            {'FuncMad', @()FuncMad(data,opts);
            'FuncMax', @()FuncMax(data,opts);
            'FuncMin', @()FuncMin(data,opts);
            'FuncKurtosis', @()FuncKurtosis(data,opts);
            'FuncIqr', @()FuncIqr(data,opts);
            'FuncAbsEnergyFreq', @()FuncAbsEnergyFreq(data,opts);
            'FuncEigPower', @()FuncEigPower(data,9,16,1.5,8,opts);
            'FuncShanEntr', @()FuncShanEntr(data,opts);
            'FuncApEntr', @()FuncApEntr(data,2,0.2,opts);
            'FuncEigTire', @()FuncEig(data,9,16,'false',opts);
            'FuncEigBody', @()FuncEig(data,0.5,4,'false',opts);
            'FuncAutocorr', @()FuncAutocorr(data,8,opts)
            }];
    end

    %% Calculate Features
    tic;
    fprintf('\nStarting calculation of features\n');
    
    numFeatures = size(calcFeatures,1);
    tmp_features = cell(numFeatures,1);

    parfor (cntFeature = 1 : numFeatures, (opts.useParallel*opts.numParallelWorkers))
        % Calculate features and save as cell array (needed for parfor)
        fprintf('Calculating feature %d of %d features\n', cntFeature, numFeatures); 
        tmp_features{cntFeature} = calcFeatures{cntFeature,2}();
    end

    % Redefine feature to struct
    features = [];
    for cntFeature = 1 : size(calcFeatures,1)
        features.(calcFeatures{cntFeature,1}) = tmp_features{cntFeature,1};
    end

    features = features2table(features,opts);

    fprintf('\nCalculation of features finished\n');
    toc

end