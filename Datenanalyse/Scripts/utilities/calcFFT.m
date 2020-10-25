function [P1] = calcFFT(x,varargin)
%CALCFFT calculates one-sided FFT (Amplitudes are already doubled)

    if isfield(x,'data')
        if ~iscell(x.data)
            x_calc = x.data(:,cnt_col);
        else
            x_calc = x.data{:,cnt_col};
        end
    else
        x_calc = x;
    end

    if size(x_calc,1)==1
        x_calc = x_calc';
    end

%     x_calc = x_calc(~isnan(x_calc));
    
    N = 2^floor(log2(size(x_calc,1)));          % Amount of sample points as power of 2
    x_calc = x_calc((1+(end-N)):end,:);            % Use last N sample points of signal
    
    H = fft(x_calc, N);
    P2 = abs(H)/(N);
    P1 = P2(1:N/2,:);
    P1(2:end,:) = 2*P2(2:N/2,:);

end

