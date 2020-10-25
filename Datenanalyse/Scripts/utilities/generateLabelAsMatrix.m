function labelAsMatrix = generateLabelAsMatrix(data, varargin)
%GENERATEFFTDATA Summary of this function goes here
%   Detailed explanation goes here

    %% Check if unique classes of exisiting classifier is available
    if find(strcmp(varargin,'uniqueClasses'))
        uniqueClasses = varargin{find(strcmp(varargin,'uniqueClasses'))+1};
    else
        uniqueClasses = [];
    end

    if isempty(uniqueClasses)
        if isfield(data.Prop, 'Label')
            uniqueClasses = unique(data.Prop.Label);
        elseif isfield(data.Prop, 'labelIsolation')
            uniqueClasses = unique(data.Prop.labelIsolation);
        else
            fprintf('no label found\n');
        end
    end
    
    %% Identify label data
    if isfield(data.Prop, 'Label')
        label = data.Prop.Label;
    elseif isfield(data.Prop, 'labelIsolation')
        label = data.Prop.labelIsolation;
    else
        fprintf('no label found\n');
    end

    %% Convert labels to matrix representation
    labelAsMatrix = zeros(size(label,1),length(uniqueClasses));
    for cntUniqueClasses = 1 : size(uniqueClasses,1)
        if iscell(uniqueClasses)
            labelAsMatrix(:,cntUniqueClasses) = strcmp(uniqueClasses{cntUniqueClasses},label);
        else
            labelAsMatrix(:,cntUniqueClasses) = strcmp(uniqueClasses(cntUniqueClasses),label);
        end
    end

end

