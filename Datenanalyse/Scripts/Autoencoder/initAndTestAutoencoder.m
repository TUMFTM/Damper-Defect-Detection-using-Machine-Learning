function features = initAndTestAutoencoder(data,mdlAutoencoder,opts,autoencoderOpts,varargin)
%INITAUTOENCODERTRAINING Summary of this function goes here
%   Detailed explanation goes here

    % Save some informations
    autoencoderOpts.numTestingObservations = size(data.(opts.fieldsData{1}),1);

    % Detrend data
    data = detrendData(data, opts);
    
    % Generate FFT
    if autoencoderOpts.useFFT
        if (~isfield(data.Prop,'isFFT')) || (~data.Prop.isFFT)  % check if data is not an FFT already
            
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
            
            data = generateFFTData(data, opts, 'windowlength', windowlength, 'windowstep', windowstep);
            
        end
    end
    
    % Calculate Autoencoder Features
    features = encodeAutoencoder(data,mdlAutoencoder(1,:),opts,autoencoderOpts);

end

