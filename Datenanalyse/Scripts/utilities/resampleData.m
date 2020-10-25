function [data, t] = resampleData(inputData, fs, varargin)
% Resamples data matrix to new sampling frequency and to uniform time
% basis
% Input matrix must contain time (column 1) and data (column 2)
% in normal cases, resampling and interp1 behave identically

    % Resampling applies a FIR filter additionally
    % Y = resample(X(:,2),X(:,1),fs);

    if find(strcmp(varargin,'method'))
        method = varargin{find(strcmp(varargin,'method'))+1};
    else
        method = 'linear';
    end
    
    fieldsTmp_data = fields(inputData);
    numFieldsTmp_Data = size(fieldsTmp_data,1);

    % find minimum and maximum time value
    tMin = zeros(numFieldsTmp_Data,1);
    tMax = zeros(numFieldsTmp_Data,1);
    for cntSens = 1 : size(fieldsTmp_data,1)
        if ~isempty(inputData.(fieldsTmp_data{cntSens}))
            tMin(cntSens) = inputData.(fieldsTmp_data{cntSens})(1,1);
            tMax(cntSens) = inputData.(fieldsTmp_data{cntSens})(end,1);
        else
            tMin(cntSens) = NaN;
            tMax(cntSens) = NaN;
        end
    end

    % Generate time vector
    t = max(tMin) : 1/fs : min(tMax);

    % Perform interpolation
    for cntSens = 1 : size(fieldsTmp_data,1)
        if ~isempty(inputData.(fieldsTmp_data{cntSens}))
            data.(fieldsTmp_data{cntSens})(:,1) = ...
                interp1(inputData.(fieldsTmp_data{cntSens})(:,1),inputData.(fieldsTmp_data{cntSens})(:,2),t,method);
        else
            data.(fieldsTmp_data{cntSens})(:,1) = NaN;
        end
    end
    
end