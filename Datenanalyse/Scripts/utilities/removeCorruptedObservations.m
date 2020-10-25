function [dataOut] = removeCorruptedObservations(dataIn, idxCorruptedObs, varargin)
%REMOVECORRUPTEDDATA Removes corrupted observations
%   Corrupted data was found manually

    if find(strcmp(varargin,'newObsID'))
        newObsID = varargin{find(strcmp(varargin,'newObsID'))+1};
    else
        newObsID = 1;
    end

    dataOut = dataIn;
    dataOut = rmfield(dataOut, 'Prop');
    fieldsData = fields(dataOut);
    for cntField = 1 : length(fieldsData)
        dataOut.(fieldsData{cntField})(idxCorruptedObs,:) = [];
    end
    dataOut.Prop = dataIn.Prop;
    fieldsProp = fields(dataOut.Prop);
    for cntField = 1 : length(fieldsProp)
        dataOut.Prop.(fieldsProp{cntField})(idxCorruptedObs) = [];
    end
    
    if newObsID
        dataOut.Prop.observationID = [1 : size(dataOut.(fieldsData{1,1}),1)]';
    end

end

