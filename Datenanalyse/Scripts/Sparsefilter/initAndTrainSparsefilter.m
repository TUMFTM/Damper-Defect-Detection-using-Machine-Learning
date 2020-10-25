function [mdlSparsefilter, sparsefilterOpts] = initAndTrainSparsefilter(dataTraining,opts,sparsefilterOpts,varargin)
%INITAUTOENCODERTRAINING Summary of this function goes here
%   Detailed explanation goes here

    % Save some informations
    sparsefilterOpts.numTrainingObservations = size(dataTraining.(opts.fieldsData{1}),1);
    sparsefilterOpts.numSegmentsPerObservation = floor(size(dataTraining.(opts.fieldsData{1}),2)/sparsefilterOpts.lengthSegment);

    % Detrend data
    dataTraining = detrendData(dataTraining, opts);
    
    % Generate FFT
    if sparsefilterOpts.useFFT
        if (~isfield(dataTraining.Prop,'isFFT')) || (~dataTraining.Prop.isFFT)  % check if data is not an FFT already
            dataTraining = generateFFTData(dataTraining, opts);
        end
    end

    % Reduce training data if required
    if sparsefilterOpts.usePassiveForTraining
        dataTrainingOrig = dataTraining;
        dataTraining = reduceDataToSpecificClass(dataTrainingOrig,'passiveIntact');
        if isempty(dataTraining.(opts.fieldsData{1}))
            dataTraining = reduceDataToSpecificClass(dataTrainingOrig,'intact');
        end
    end
    
    % Train Sparsefilter
    tmpmdl = trainSparsefilter(dataTraining,opts,sparsefilterOpts);
    if iscell(tmpmdl)
        mdlSparsefilter(1,:) = tmpmdl;
    else
        mdlSparsefilter{1,:} = tmpmdl;
    end

end

