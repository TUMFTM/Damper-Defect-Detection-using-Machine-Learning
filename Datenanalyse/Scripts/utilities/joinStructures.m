function [struct1] = joinStructures(struct1,identifier1,struct2,identifier2)
    % Check input
    if nargin<4
        error('joinStructures:numberOfInputs','Wrong number of inputs.')
    elseif ~isstruct(struct1)
        error('joinStructures:wrongInput','First input must be a structure.')
    elseif ~isfield(struct1, identifier1)
        error('joinStructures:wrongInput', sprintf('''%s'' must be field of first structure.',identifier1))
    elseif ~isstruct(struct2)
        error('joinStructures:wrongInput','Third input must be a structure.')        
    elseif ~isfield(struct2, identifier2)
        error('joinStructures:wrongInput', sprintf('''%s'' must be field of second structure.',identifier2))
    end
    % Check fieldnames
    fields1 = fieldnames(struct1);
    fields2 = fieldnames(struct2);
    if length(fields1)~=length(fields2), error('joinStructures:wrongInput', 'Structures have different fields.'), end
    for iField=1:length(fields1)
        if ~any(~cellfun(@isempty, regexp(fields2, ['^',fields1{iField},'$'])))
            error('joinStructures:wrongInput', sprintf('Field %s is not in seconde structure.',fields1{iField}))
        end
    end
        
    % Get Intersection and add field at end
    [indexIntersect] = ismember(struct2.(identifier1),struct1.(identifier2));
	if any(~indexIntersect)
        for iField=1:length(fields1)
            try
            struct1.(fields1{iField})(end+1:end+sum(~indexIntersect)) = struct2.(fields1{iField})(~indexIntersect);
            catch ME
                rethrow(ME)
            end
        end
	end
end

