function Y = bit2freq(x,fs)
% Transforms bit number (x) to frequency (Y) for 128bit signal

x_norm = 128;
Y = x*fs/(2*(x_norm+1));

end

