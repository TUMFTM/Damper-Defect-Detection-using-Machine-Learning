function dataOut = reduceDataToSpecificClass(dataStruct,className,varargin)
%REDUCEDATATOSPECIFICCLASS Summary of this function goes here
%   If multiple classes are given in className, an OR-combination is
%   performed

    if find(strcmp(varargin,'idxSelectedClass'))
        idxSelectedClass = varargin{find(strcmp(varargin,'idxSelectedClass'))+1};
    else
        idxSelectedClass = [];
    end

    dataOut = dataStruct;
    fieldsData = fields(dataOut);
    fieldsData(strcmp('Prop',fieldsData)) = [];
    
    if isempty(idxSelectedClass)
        if iscell(className)
            % Perform OR-combination of multiple selected classes
            idxSelectedClass = false(size(dataOut.Prop.labelIsolation));
            for cntClassNames = 1 : length(className)
                newSelectedClass = strcmp(dataOut.Prop.labelIsolation,className{cntClassNames});
                idxSelectedClass = idxSelectedClass | newSelectedClass;
                if max(newSelectedClass)==0
                    fprintf('Class %s not found in dataStruct during reduction to specific class\n', className{cntClassNames});
                end
            end
        else
            idxSelectedClass = strcmp(dataOut.Prop.labelIsolation,className);
        end
    end
    
    for cntFields = 1 : length(fieldsData)
        dataOut.(fieldsData{cntFields})(~idxSelectedClass,:) = [];
    end
    
    if isfield(dataOut,'Prop')
        fieldsProp = fields(dataOut.Prop);
        for cntFields = 1 : length(fieldsProp)
            if length(dataOut.Prop.(fieldsProp{cntFields})) > 1
                dataOut.Prop.(fieldsProp{cntFields})(~idxSelectedClass,:) = [];
            end
        end
    end
end

