function autoenc = trainMyAutoencoder(X, varargin)
% trainAutoencoder   Train an autoencoder
%   autoenc = trainAutoencoder(x) returns a trained autoencoder where x is 
%   the training data.
%
%   x may be a matrix of training samples where each column represents a
%   single sample, or alternatively it can be a cell array of images, where
%   each image has the same number of dimensions.
%
%   autoenc = trainAutoencoder(x, hiddenSize) returns a trained autoencoder
%   where x is the training data, and hiddenSize specifies the size of the 
%   autoencoder's hidden representation. The default value of hiddenSize is
%   10.
%
%   autoenc = trainAutoencoder(..., Name1, Value1, Name2, Value2, ...)
%   returns a trained autoencoder with additional options specified by the
%   following name/value pairs:
%
%       'EncoderTransferFunction' - The transfer function for the encoder.
%                                   This can be either 'logsig' for the
%                                   logistic sigmoid function, or 'satlin'
%                                   for the positive saturating linear
%                                   transfer function. The default is
%                                   'logsig'.
%       'DecoderTransferFunction' - The transfer function for the decoder.
%                                   This can be 'logsig' for the logistic
%                                   sigmoid function, 'satlin' for the
%                                   positive saturating linear transfer
%                                   function, or 'purelin' for the linear
%                                   transfer function. The default is
%                                   'logsig'.
%       'MaxEpochs'               - The maximum number of training epochs. 
%                                   The default is 1000.
%       'L2WeightRegularization'  - The coefficient that controls the
%                                   weighting of the L2 weight regularizer.
%                                   The default value is 0.001.
%       'LossFunction'            - The loss function that is used for
%                                   training. The default is 'msesparse'.
%       'ShowProgressWindow'      - Indicates whether the training window 
%                                   should be shown during training. The 
%                                   default is true.
%       'SparsityProportion'      - The desired proportion of training
%                                   examples which a neuron in the hidden 
%                                   layer of the autoencoder should 
%                                   activate in response to. Must be 
%                                   between 0 and 1. A low value encourages
%                                   a higher degree of sparsity. The 
%                                   default is 0.05.
%       'SparsityRegularization'  - The coefficient that controls the
%                                   weighting of the sparsity regularizer.
%                                   The default is 1.
%       'TrainingAlgorithm'       - The training algorithm used to train
%                                   the autoencoder. Only the value
%                                   'trainscg' for scaled conjugate
%                                   gradient descent is allowed, which is
%                                   the default.
%       'ScaleData'               - True when the autoencoder rescales the 
%                                   input data. The default is true.
%       'UseGPU'                  - True if the GPU is used for training.
%                                   The default value is false.
%
%   Example 1: 
%       Train a sparse autoencoder to compress and reconstruct abalone
%       shell ring data, and then measure the mean squared reconstruction
%       error.
%
%       x = abalone_dataset;
%       hiddenSize = 4;
%       autoenc = trainAutoencoder(x, hiddenSize, ...
%           'MaxEpochs', 400, ...
%           'DecoderTransferFunction', 'purelin');
%       xReconstructed = predict(autoenc, x);
%       mseError = mse(x-xReconstructed)
%
%   Example 2:
%       Train a sparse autoencoder on images of handwritten digits to learn
%       features, and use it to compress and reconstruct these images. View
%       some of the original images along with their reconstructed
%       versions.
%
%       x = digitSmallCellArrayData;
%       hiddenSize = 40;
%       autoenc = trainAutoencoder(x, hiddenSize, ...
%           'L2WeightRegularization', 0.004, ...
%           'SparsityRegularization', 4, ...
%           'SparsityProportion', 0.15);
%       xReconstructed = predict(autoenc, x);
%       figure;
%       for i = 1:20
%           subplot(4,5,i);
%           imshow(x{i});
%       end
%       figure;
%       for i = 1:20
%           subplot(4,5,i);
%           imshow(xReconstructed{i});
%       end
%
%   See also Autoencoder

%   Copyright 2015-2016 The MathWorks, Inc.

paramsStruct  = MyAutoencoder.parseInputArguments(varargin{:});
autonet = MyAutoencoder.createNetwork(paramsStruct);

% autonet.trainParam.min_grad
% autonet.trainParam.showCommandLine = true;
% autonet.trainParam.lambda = 5e-7;


autoenc = MyAutoencoder.train(X, autonet, paramsStruct.UseGPU);
end