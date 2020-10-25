function [mdlAutoencoder, autoencoderOpts] = initAndTrainAutoencoder(dataTraining,opts,autoencoderOpts,varargin)
%INITAUTOENCODERTRAINING Summary of this function goes here
%   Detailed explanation goes here

    % Save some informations
    autoencoderOpts.numTrainingObservations = size(dataTraining.(opts.fieldsData{1}),1);
    autoencoderOpts.numSegmentsPerObservation = floor(size(dataTraining.(opts.fieldsData{1}),2)/autoencoderOpts.lengthSegment);
    
    % Detrend data
    dataTraining = detrendData(dataTraining, opts);
    
    % Generate FFT
    if autoencoderOpts.useFFT
        if (~isfield(dataTraining.Prop,'isFFT')) || (~dataTraining.Prop.isFFT)  % check if data is not an FFT already
            
            if isfield(autoencoderOpts, 'windowlength')
                windowlength = autoencoderOpts.windowlength;
            else
                windowlength = 0; % no windowing
            end
            
            if isfield(autoencoderOpts, 'windowstep')
                windowstep = autoencoderOpts.windowstep;
            else
                windowstep = windowlength; % no overlap
            end
            
            dataTraining = generateFFTData(dataTraining, opts, 'windowlength', windowlength, 'windowstep', windowstep);
        end
    end

    % Reduce training data if required
    if autoencoderOpts.usePassiveForTraining
        dataTrainingOrig = dataTraining;
        dataTraining = reduceDataToSpecificClass(dataTrainingOrig,'passiveIntact');
        if isempty(dataTraining.(opts.fieldsData{1}))
            dataTraining = reduceDataToSpecificClass(dataTrainingOrig,'intact');
        end
    end
    
    % Train Autoencoder
    tmpmdlAutoencoder = trainAutoencoder(dataTraining,opts,autoencoderOpts,'optimizeHyperparameter',autoencoderOpts.optimizeHyperparameter);
    if iscell(tmpmdlAutoencoder)
        mdlAutoencoder(1,:) = tmpmdlAutoencoder;
    else
        mdlAutoencoder{1,:} = tmpmdlAutoencoder;
    end

end

