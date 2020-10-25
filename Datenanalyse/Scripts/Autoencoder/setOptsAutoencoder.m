function autoencoderOpts = setOptsAutoencoder(varargin)
%SETOPTSAUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

    autoencoderOpts = struct;

    if find(strcmp(varargin,'usePassiveForTraining'))
        autoencoderOpts.usePassiveForTraining = varargin{find(strcmp(varargin,'usePassiveForTraining'))+1};
    else
        autoencoderOpts.usePassiveForTraining = 0;
    end
    
    autoencoderOpts.useFFT = 1;
    autoencoderOpts.windowlength = 128;
    autoencoderOpts.windowstep = 64;
    
    autoencoderOpts.optimizeHyperparameter = 1; % 1 = optimize hyperparameters of Autoencoder
    
    autoencoderOpts.lengthSegment = 0;
    autoencoderOpts.numFeatures = [50];
    autoencoderOpts.AutoencoderType = 'OneIndividualAutoencoderForEachSignal';
    % 'OneIndividualAutoencoderForEachSignal'
    % 'OneSingleAutoencoderForAllAppendedSignals'
    % 'OneSingleAutoencoderForAllSignals'
    % 'OneSingleAutoencoderForWheelSpeedsAndOneIndividualAutoencoderForRest'

end

