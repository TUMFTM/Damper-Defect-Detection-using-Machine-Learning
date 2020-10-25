function Y = freq2bit(x,fs)
% Transforms frequency (x) to bit number (Y) for a 128bit signal

x_norm = 128;
Y = floor(2*x*(x_norm+1)/fs);

end

