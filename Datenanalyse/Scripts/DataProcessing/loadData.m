function data = loadData(filename, foldername, opts, varargin)
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here

    fprintf('Loading data...\n');
    
    % set control variable
    if find(strcmp(varargin,'publication'))
        ctrl.publication = varargin{find(strcmp(varargin,'publication'))+1};
    else
        ctrl.publication = 0;
    end
    
    %% Get labeling data
    if iscell(filename)     % if-cause and for-loop is needed for MultiSelect
        Label_Data = table;
        for cntLabelFiles = 1 : length(filename)
            Label_Data = [Label_Data; readtable([foldername, filename{cntLabelFiles}])];
        end
    else
        Label_Data = readtable([foldername, filename]);
    end
    fprintf('Load labeling data successful.\n');


    %% Load Data
    data = inputMeasurement(Label_Data,opts, 'publication', ctrl.publication);
    
    %% Add unique ID for each observation
    if isfield(data, 'Prop')
        data.Prop.observationID = [1 : size(data.(opts.fieldsData{1,1}),1)]';
    end
    
    fprintf('Loading data...finished\n');
end

