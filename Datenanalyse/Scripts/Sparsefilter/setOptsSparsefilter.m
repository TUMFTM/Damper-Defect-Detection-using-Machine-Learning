function sparsefilterOpts = setOptsSparsefilter(varargin)
%SETOPTSAUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

    sparsefilterOpts = struct;

    if find(strcmp(varargin,'usePassiveForTraining'))
        sparsefilterOpts.usePassiveForTraining = varargin{find(strcmp(varargin,'usePassiveForTraining'))+1};
    else
        sparsefilterOpts.usePassiveForTraining = 0;
    end
    
    sparsefilterOpts.useFFT = 0;
    
    sparsefilterOpts.lengthSegment = 64;    % use integer divider of selected observation length (e.g. observation length = 512 samples -> use 2, 4, 8, 16 etc.)
    sparsefilterOpts.numFeatures = 50;
    sparsefilterOpts.Type = 'OneIndividualSparsefilterForEachSignal';
    % 'OneIndividualSparsefilterForEachSignal'
    % 'OneSingleSparsefilterForAllAppendedSignals'
    % 'OneSingleSparsefilterForAllSignals'
    % 'OneSingleSparsefilterForWheelSpeedsAndOneIndividualSparsefilterForRest'

end

