function features = initAndTestSparsefilter(data,mdlSparsefilter,opts,sparsefilterOpts,varargin)
%INITANDTESTSPARSEFILTER Summary of this function goes here
%   Detailed explanation goes here

    % Save some informations
    sparsefilterOpts.numTestingObservations = size(data.(opts.fieldsData{1}),1);

    % Detrend data
    data = detrendData(data, opts);
    
    % Generate FFT
    if sparsefilterOpts.useFFT
        if (~isfield(data.Prop,'isFFT')) || (~data.Prop.isFFT)  % check if data is not an FFT already
            data = generateFFTData(data, opts);
        end
    end
    
    % Calculate Features
    features = encodeSparsefilter(data,mdlSparsefilter(1,:),opts,sparsefilterOpts);

end

