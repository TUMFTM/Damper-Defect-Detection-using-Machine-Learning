function fftData = generateFFTDataAsTable(x,opts, varargin)
%GENERATEFFTDATAASTABLE Generates FFT as table with Properties

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

    % Detrend data
    x = detrendData(x, opts);
    
    fftData = generateFFTData(x, opts, 'windowlength', windowlength, 'windowstep', windowstep);
    Prop = fftData.Prop;
    fftData = rmfield(fftData, 'Prop');
    fftData = struct2table(fftData);

end

