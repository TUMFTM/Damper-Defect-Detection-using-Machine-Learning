function [mdlStackedAutoencoder] = trainStackedAutoencoder(dataTraining,sizeHiddenLayer,varargin)
%calcStackedAutoencoderWithSoftmaxLayer Summary of this function goes here
%   Detailed explanation goes here

    %% Training
  
    if find(strcmp(varargin,'optimizeHyperparameter'))
        optimizeHyperparameter = varargin{find(strcmp(varargin,'optimizeHyperparameter'))+1};
    else
        optimizeHyperparameter = 0;
    end
    
    if optimizeHyperparameter
        statusFolder = mkdir('Autoencoder/resultsOptim');
        if ~statusFolder
            % Create folder iteratively
            currPath = pwd;
            for pathPart = split('Autoencoder/resultsOptim')
                if ~exist(pathPart{1},'dir')
                    mkdir(pathPart{1});
                end
                cd(pathPart{1});
            end
            cd(currPath);
        end
    end
    
    % delete layers with size 0
    sizeHiddenLayer = sizeHiddenLayer(~sizeHiddenLayer==0);
    
    mdlAutoencoder = cell(size(sizeHiddenLayer));
    outputFeaturesOfAutoencoder = cell(size(sizeHiddenLayer));
    for cntHiddenLayer = 1 : length(sizeHiddenLayer)
        if cntHiddenLayer > 1
            inputToAutoencoder = outputFeaturesOfAutoencoder{cntHiddenLayer-1}';
        else
            inputToAutoencoder = dataTraining';
        end
        
        %% Perform Bayesian Optimization
        if optimizeHyperparameter == 1
            optimVars = [
                optimizableVariable('L2WeightRegularization',[1e-18 1e-4],'Transform','log')
                optimizableVariable('SparsityRegularization',[1e-12 1e-4],'Transform','log')];
            
            ObjFcn = makeObjFcn(inputToAutoencoder,sizeHiddenLayer(cntHiddenLayer));
            BayesObject = bayesopt(ObjFcn,optimVars,...
                'MaxObj',100,...
                'IsObjectiveDeterministic',false,...
                'UseParallel',true);    % use parallel if no pool is running
            bestIdx = BayesObject.IndexOfMinimumTrace(end);
            fileName = BayesObject.UserDataTrace{bestIdx};
            load(fileName);
            
            mdlAutoencoder{cntHiddenLayer} = tmpmdlAutoencoder;

        else
            try
                mdlAutoencoder{cntHiddenLayer} = trainMyAutoencoder(inputToAutoencoder,sizeHiddenLayer(cntHiddenLayer),...
                    'MaxEpochs',500,'ShowProgressWindow',true,'SparsityProportion',0.01,...
                    'L2WeightRegularization',1e-15,'SparsityRegularization',1e-6,...
                    'MinGradient',1e-12,...
                    'ScaleData',true,'UseGPU',true);
                fprintf('used GPU\n');
            catch
                mdlAutoencoder{cntHiddenLayer} = trainMyAutoencoder(inputToAutoencoder,sizeHiddenLayer(cntHiddenLayer),...
                    'MaxEpochs',500,'ShowProgressWindow',true,'SparsityProportion',0.01,...
                    'L2WeightRegularization',1e-15,'SparsityRegularization',1e-6,...
                    'MinGradient',1e-12,...
                    'ScaleData',true);
                fprintf('used CPU\n');
            end
        end
        outputFeaturesOfAutoencoder{cntHiddenLayer} = encode(mdlAutoencoder{cntHiddenLayer},inputToAutoencoder)';
    end

    % Stack all Autoencoder
    mdlStackedAutoencoder = mdlAutoencoder{1};
    for cntHiddenLayer = 2 : length(sizeHiddenLayer)
        mdlStackedAutoencoder = stack(mdlStackedAutoencoder,mdlAutoencoder{cntHiddenLayer});
    end
    
end



function ObjFcn = makeObjFcn(inputToAutoencoder,sizeHiddenLayer)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        
        n = size(inputToAutoencoder,2);
        tf = false(1,n);
        tf(1:round(0.8*n)) = true;
        tf = tf(randperm(n));
        
        try
            tmpmdlAutoencoder = trainMyAutoencoder(inputToAutoencoder(:,tf),sizeHiddenLayer,...
                    'MaxEpochs',300,'ShowProgressWindow',true,'SparsityProportion',0.01,...
                    'L2WeightRegularization',optVars.L2WeightRegularization,'SparsityRegularization',optVars.SparsityRegularization,...
                    'MinGradient',1e-12,...
                    'ScaleData',true,'UseGPU',true);
                fprintf('used GPU\n');
        catch
            tmpmdlAutoencoder = trainMyAutoencoder(inputToAutoencoder(:,tf),sizeHiddenLayer,...
                    'MaxEpochs',300,'ShowProgressWindow',true,'SparsityProportion',0.01,...
                    'L2WeightRegularization',optVars.L2WeightRegularization,'SparsityRegularization',optVars.SparsityRegularization,...
                    'MinGradient',1e-12,...
                    'ScaleData',true);
                fprintf('used CPU\n');
        end
        valError = sum(sum((inputToAutoencoder(:,~tf) - predict(tmpmdlAutoencoder,inputToAutoencoder(:,~tf))).^2));
        
        %%
        timeAsString = datestr(now(),'yymmdd_HHMMSS');
        fileName = "Autoencoder/resultsOptim/" + timeAsString + "_" + num2str(valError,10) + ".mat";
        statusFolder = mkdir('Autoencoder/resultsOptim');
        if ~statusFolder
            % Create folder iteratively
            currPath = pwd;
            for pathPart = split('Autoencoder/resultsOptim')
                if ~exist(pathPart{1},'dir')
                    mkdir(pathPart{1});
                end
                cd(pathPart{1});
            end
            cd(currPath);
        end
        save(fileName,'tmpmdlAutoencoder','valError')
        cons = [];
    end
end
        