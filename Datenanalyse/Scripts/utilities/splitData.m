function [dataTraining,dataTesting] = splitData(data,tf,opts)
%SPLITDATA Summary of this function goes here
%   Detailed explanation goes here

    % Split data to trainings and test data
    tmpData = rmfield(data,'Prop');
    dataTraining = opts.data;
    dataTesting = opts.data;
    
    dataTraining = rmfield(dataTraining,'Prop');
    dataTesting = rmfield(dataTesting,'Prop');
    
    %% Divide data
    if isempty(tf)
        n = size(data.(opts.fieldsData{1}),1);
        tf = false(n,1);
        tf(1:round(opts.splitTrainingData*n)) = true;
        tf = tf(randperm(n));
    end
    
    for cntFields = 1 : size(opts.fieldsData,1)
        dataTraining.(opts.fieldsData{cntFields}) = tmpData.(opts.fieldsData{cntFields})(tf,:);
        dataTesting.(opts.fieldsData{cntFields}) = tmpData.(opts.fieldsData{cntFields})(~tf,:);
    end

    fieldsProp = fields(data.Prop);
    for cntFieldsProp = 1 : length(fieldsProp)
        if size(data.Prop.(fieldsProp{cntFieldsProp}),1) == size(tf,1)
            dataTraining.Prop.(fieldsProp{cntFieldsProp}) = data.Prop.(fieldsProp{cntFieldsProp})(tf,:);
            dataTesting.Prop.(fieldsProp{cntFieldsProp}) = data.Prop.(fieldsProp{cntFieldsProp})(~tf,:);
        else
            dataTraining.Prop.(fieldsProp{cntFieldsProp}) = data.Prop.(fieldsProp{cntFieldsProp});
            dataTesting.Prop.(fieldsProp{cntFieldsProp}) = data.Prop.(fieldsProp{cntFieldsProp});
        end
    end

end
