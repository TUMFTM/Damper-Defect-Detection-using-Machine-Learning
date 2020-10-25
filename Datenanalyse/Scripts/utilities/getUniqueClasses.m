function uniqueClasses = getUniqueClasses(data)
%GENERATEFFTDATA Summary of this function goes here
%   Detailed explanation goes here

    try
        uniqueClasses = unique(data.Prop.Label);
    catch 
        uniqueClasses = unique(data.Prop.labelIsolation);
    end

end

