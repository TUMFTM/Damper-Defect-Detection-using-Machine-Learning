function [fftData] = generateFFTData(data, opts, varargin)
%GENERATEFFTDATA Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'windowlength'))
        windowlength = varargin{find(strcmp(varargin,'windowlength'))+1};
    else
        windowlength = 0;
    end
    
    if find(strcmp(varargin,'windowstep'))
        windowstep = varargin{find(strcmp(varargin,'windowstep'))+1};
    else
        windowstep = windowlength;
    end

    % Calculate fft for all data
    fftData = opts.data;
    fftData = rmfield(fftData,'Prop');
    for cntFields = 1 : size(opts.fieldsData,1)
        
        % Reshape input if segmentation is required
        if windowlength > 0
            tmpData = data.(opts.fieldsData{cntFields});
            [nrows, ncols] = size(tmpData);
            segmented_data = tmpData(bsxfun(@plus, ...
                                  bsxfun(@plus, ...
                                         1:windowlength, ...
                                         permute(0:windowstep:ncols-windowlength, [1 3 2])) * nrows, ...
                                  (1-nrows:0).'));
            inputToCalcFFT = reshape(permute(segmented_data,[3,1,2]),[],windowlength);
        else
            inputToCalcFFT = data.(opts.fieldsData{cntFields});
        end
        
        % Calculation of FFT
        calculatedFFTdata = calcFFT(inputToCalcFFT')';
        
        % Calculate mean if segmentation is required
        if windowlength > 0
            meanFFTdata = zeros(size(data.(opts.fieldsData{1}),1),size(calculatedFFTdata,2));
            numSegmentsPerObservation = size(segmented_data,3);
            for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                meanFFTdata(cntObs,:) = mean(calculatedFFTdata((cntObs-1)*numSegmentsPerObservation+1:cntObs*numSegmentsPerObservation,:),1);
            end
            fftData.(opts.fieldsData{cntFields}) = meanFFTdata;
        else
            fftData.(opts.fieldsData{cntFields}) = calculatedFFTdata;
        end
    end
    
    if isstruct(data)
        if isfield(data,'Prop')
            fftData.Prop = data.Prop;
            % set variable that shows that data is already fft'ed
            fftData.Prop.isFFT = 1;  
        end
    elseif istable(data)
        fftData = struct2table(fftData);
        if max(strcmp(fields(data),'Label'))
            fftData.Label = data.Label;
        end
    end
    
end

