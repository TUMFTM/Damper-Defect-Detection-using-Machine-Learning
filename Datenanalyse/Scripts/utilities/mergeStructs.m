function [output] = mergeStructs(structA,structB,dim)
%MERGESTRUCTS Merges two structs
% If fields exist in both structs -> Concatenate fields
% if field exist only in one struct -> add it to output struct
% dim = 1 -> append row-wise
% dim = 2 -> append column-wise

    output = structA;
    fieldsA = fields(structA);
    fieldsB = fields(structB);
    
    for cntFieldsA = 1 : length(fieldsA)
        if isfield(structB, fieldsA{cntFieldsA})
            % aktuelles Feld ist Teil von structB
            if max(size(structA.(fieldsA{cntFieldsA}))) > 1
                % aktuelles Feld hat Vektor oder Matrix Größe
                if iscell(structA.(fieldsA{cntFieldsA}))
                    % Feld is cell
                    if dim == 1
                        output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}); structB.(fieldsB{cntFieldsA})];
                    elseif dim == 2
                        output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}), structB.(fieldsB{cntFieldsA})];
                    end
                elseif isstruct(structA.(fieldsA{cntFieldsA}))
                    % Feld ist struct
                    output.(fieldsA{cntFieldsA}) = mergeStructs(structA.(fieldsA{cntFieldsA}),structB.(fieldsB{cntFieldsA}));
                elseif istable(structA.(fieldsA{cntFieldsA}))
                    % Feld ist table
                    if dim == 1
                        output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}); structB.(fieldsB{cntFieldsA})];
                    elseif dim == 2
                        output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}), structB.(fieldsB{cntFieldsA})];
                    end
                else
                    % Feld ist irgendetwas anderes (kein cell, struct oder table)
                    if ischar(structA.(fieldsA{cntFieldsA}))
                        output.(fieldsA{cntFieldsA}) = {};
                        output.(fieldsA{cntFieldsA}){1,1} = structA.(fieldsA{cntFieldsA});
                        output.(fieldsA{cntFieldsA}){2,1} = structB.(fieldsB{cntFieldsA});
                    else
                        output.(fieldsA{cntFieldsA}) = cat(dim,structA.(fieldsA{cntFieldsA}),structB.(fieldsB{cntFieldsA}));
                    end
                end
            else
                % aktuelles Feld ist nur ein Element
                if isequal(structA.(fieldsA{cntFieldsA}), structB.(fieldsA{cntFieldsA}))
                    % aktuelles Element is identisch in beiden structs
                    continue;
                else
                    % aktuelles Element ist unterschiedlich in beiden structs
                    if isstruct(structA.(fieldsA{cntFieldsA}))
                        % aktuelles Feld ist ein struct
                        output.(fieldsA{cntFieldsA}) = mergeStructs(structA.(fieldsA{cntFieldsA}),structB.(fieldsB{cntFieldsA}),1);
                    else
                        if iscell(structA.(fieldsA{cntFieldsA}))
                            output.(fieldsA{cntFieldsA}){1,1} = structA.(fieldsA{cntFieldsA});
                            output.(fieldsA{cntFieldsA}){2,1} = structB.(fieldsA{cntFieldsA});
                        else
                            if dim == 1
                                output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}); structB.(fieldsB{cntFieldsA})];
                            elseif dim == 2
                                output.(fieldsA{cntFieldsA}) = [structA.(fieldsA{cntFieldsA}), structB.(fieldsB{cntFieldsA})];
                            end
                        end
                    end
                end
            end
        end
    end

end

