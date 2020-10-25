function [balancedData, restData] = balanceDataset(data, opts)
%BALANCEDATASET Summary of this function goes here
%   Identify class with least number of observations and reduce all other
%   classes to this number of observations randomly
%   postprocessing of probabilites might be more difficult with this method
%   of balancing a dataset

    balancedData = opts.data;
    restData = opts.data;
    classNames = unique(data.Prop.labelIsolation);
    
    fieldsData = fields(data);
    fieldsData(strcmp('Prop',fieldsData)) = [];
    idxSelectedClass = zeros(size(data.(fieldsData{1}),1),length(classNames));
    
    for cntClass = 1 : length(classNames)
        idxSelectedClass(:,cntClass) = strcmp(data.Prop.labelIsolation,classNames{cntClass});
    end
    
    numObsPerClass = sum(idxSelectedClass);
    minNumObsPerClass = min(numObsPerClass);
    
    %% Initialize fields
    % Copy data of selected observations
    for cntFields = 1 : size(opts.fieldsData,1)
        balancedData.(opts.fieldsData{cntFields}) = zeros(length(classNames)*minNumObsPerClass,size(data.(opts.fieldsData{cntFields}),2));
        restData.(opts.fieldsData{cntFields}) = zeros(sum(numObsPerClass)-length(classNames)*minNumObsPerClass,size(data.(opts.fieldsData{cntFields}),2));
    end

    % Copy properties of selected observations
    fieldsProp = fields(data.Prop);
    for cntFieldsProp = 1 : length(fieldsProp)
        balancedData.Prop.(fieldsProp{cntFieldsProp}) = []; % empty because datatype may be different from numeric values (e. g. datetime -> leads to errors)
        restData.Prop.(fieldsProp{cntFieldsProp}) = [];     % empty because datatype may be different from numeric values (e. g. datetime -> leads to errors)
    end
    
    %% Copy selected observations
    for cntClass = 1 : length(classNames)
        
        rng default;    % used to select consistenly random
        tf = false(numObsPerClass(cntClass),1);
        tf(1:minNumObsPerClass) = true;
        tf = tf(randperm(numObsPerClass(cntClass)));
        
        dataOfSpecificClass = reduceDataToSpecificClass(data,classNames{cntClass});
        
        % Copy data of selected observations
        for cntFields = 1 : size(opts.fieldsData,1)
            balancedData.(opts.fieldsData{cntFields})((cntClass-1)*minNumObsPerClass+1:cntClass*minNumObsPerClass,:) = dataOfSpecificClass.(opts.fieldsData{cntFields})(tf,:);
            restData.(opts.fieldsData{cntFields})((sum(numObsPerClass(1:cntClass-1))-(cntClass-1)*minNumObsPerClass)+1:(sum(numObsPerClass(1:cntClass))-cntClass*minNumObsPerClass),:) = dataOfSpecificClass.(opts.fieldsData{cntFields})(~tf,:);
        end

        % Copy properties of selected observations
        for cntFieldsProp = 1 : length(fieldsProp)
            balancedData.Prop.(fieldsProp{cntFieldsProp}) = [balancedData.Prop.(fieldsProp{cntFieldsProp}); dataOfSpecificClass.Prop.(fieldsProp{cntFieldsProp})(tf,:)];
            restData.Prop.(fieldsProp{cntFieldsProp}) = [restData.Prop.(fieldsProp{cntFieldsProp}); dataOfSpecificClass.Prop.(fieldsProp{cntFieldsProp})(~tf,:)];
        end
        
    end
        
end

