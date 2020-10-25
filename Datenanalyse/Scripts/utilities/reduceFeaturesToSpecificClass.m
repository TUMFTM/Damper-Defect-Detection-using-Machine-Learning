function dataOut = reduceFeaturesToSpecificClass(dataIn,className,varargin)
%REDUCEFEATURESTOSPECIFICCLASS Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'idxSelectedClass'))
        idxSelectedClass = varargin{find(strcmp(varargin,'idxSelectedClass'))+1};
    else
        idxSelectedClass = [];
    end

    

    %% get indizes of selected observations
    if isempty(idxSelectedClass)
        
        % Find labelIsolation-field in struct
        if isfield(dataIn,'Prop')
            locallabelIsolation = dataIn.Prop.labelIsolation;
        end
        if isfield(dataIn,'features') && ~exist('locallabelIsolation','var')
            if isfield(dataIn.features,'Prop')
                locallabelIsolation = dataIn.features.Prop.labelIsolation;
            end
        end
        if isfield(dataIn,'data') && ~exist('locallabelIsolation','var')
            if isfield(dataIn.data,'Prop')
                locallabelIsolation = dataIn.data.Prop.labelIsolation;
            end
        end
        
        if iscell(className)
            % Perform OR-combination of multiple selected classes
            idxSelectedClass = false(size(locallabelIsolation));
            for cntClassNames = 1 : length(className)
                newSelectedClass = strcmp(locallabelIsolation,className{cntClassNames});
                idxSelectedClass = idxSelectedClass | newSelectedClass;
                if max(newSelectedClass)==0
                    fprintf('Class %s not found in dataStruct during reduction to specific class\n', className{cntClassNames});
                end
            end
        else
            idxSelectedClass = strcmp(locallabelIsolation,className);
        end
    end
    
    %% Perform reduction to selected class
    dataOut = dataIn;
    fieldsData = fields(dataOut);

    for cntFields = 1 : length(fieldsData)
        if isstruct(dataOut.(fieldsData{cntFields}))
            dataOut.(fieldsData{cntFields}) = reduceFeaturesToSpecificClass(dataIn.(fieldsData{cntFields}),className,'idxSelectedClass',idxSelectedClass);
        else
            if isequal(length(idxSelectedClass),size(dataOut.(fieldsData{cntFields}),1))
                dataOut.(fieldsData{cntFields})(~idxSelectedClass,:) = [];
            else
                dataOut.(fieldsData{cntFields}) = dataIn.(fieldsData{cntFields});
                if strcmp(fieldsData{cntFields},'uniqueClasses')
                    dataOut.(fieldsData{cntFields}) = className;
                end
            end
        end
    end
    
    if max(strcmp(fieldsData,'labelAsMatrix'))
        dataOut.labelAsMatrix = generateLabelAsMatrix(dataOut);
    end
    

end

