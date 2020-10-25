function dataOut = reduceDataWithProperty(dataStruct,propertyName,functionHandle)
%reduceDataWithProperty Deletes observations with a specific property
% Inputs:
% dataStruct = dataTesting
% propertyName = 'meanAx'
% functionHandle = @(x)x<=0.5
% dataTestingReduced = reduceDataWithProperty(dataTesting,propertyName,functionHandle)


    dataOut = dataStruct;
    fieldsData = fields(dataOut);
    fieldsData(strcmp('Prop',fieldsData)) = [];
    idxSelectedObs = functionHandle(dataStruct.Prop.(propertyName));
    for cntFields = 1 : length(fieldsData)
        dataOut.(fieldsData{cntFields})(~idxSelectedObs,:) = [];
    end
    
    fieldsProp = fields(dataOut.Prop);
    for cntFields = 1 : length(fieldsProp)
        if isstruct(dataOut.Prop.(fieldsProp{cntFields}))
            for subFieldProp = fieldnames(dataOut.Prop.(fieldsProp{cntFields}))
                if length(dataOut.Prop.(fieldsProp{cntFields}).(subFieldProp{1}))==1
                    % no reduction if field is just one value (e. g.
                    % Prop.isFFT = 1
                    continue
                end
                dataOut.Prop.(fieldsProp{cntFields}).(subFieldProp{1})(~idxSelectedObs,:) = [];
            end
        else
            if length(dataOut.Prop.(fieldsProp{cntFields})) == 1
                % no reduction if field is just one value (e. g. Prop.isFFT = 1)
                continue
            end
            dataOut.Prop.(fieldsProp{cntFields})(~idxSelectedObs,:) = [];
        end
    end
end

