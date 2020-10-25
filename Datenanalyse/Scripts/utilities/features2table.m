function Y = features2table(features,opts)
%FEATURES2TABLE_ Summary of this function goes here
%   Detailed explanation goes here
    
    if isfield(features,'Prop')
        featuresProp = features.Prop;
        features = rmfield(features,'Prop');
    end

    Y = table;
    tmp_Y = table;
    fieldsFeatures = fields(features);
    for cntFeature = 1 : size(fieldsFeatures,1)

        nameFeature = fieldsFeatures{cntFeature};

        fieldsFeatureSub = fields(features.(fieldsFeatures{cntFeature}));


        if isstruct(features.(fieldsFeatures{cntFeature}).(fieldsFeatureSub{1}))

            % if next layer is still a struct
            for cntSubFeature = 1 : size(fieldsFeatureSub,1)

                nameSubFeature = [nameFeature,'_',fieldsFeatureSub{cntSubFeature}];
                tmp_Y = struct2table(features.(fieldsFeatures{cntFeature}).(fieldsFeatureSub{cntSubFeature}));

                fieldsFeatureSubSub = fields(features.(fieldsFeatures{cntFeature}).(fieldsFeatureSub{cntSubFeature}));

                % Redefine table column names
                for cntSubSubFeature = 1 : size(fieldsFeatureSubSub,1)
                    tmp_Y.Properties.VariableNames{cntSubSubFeature} = [nameSubFeature,'_', tmp_Y.Properties.VariableNames{cntSubSubFeature}];
                end

                % Append subfeatures to overall Y-variable
                Y = [Y, tmp_Y];
            end

        else

            % if next layer is not a struct
            tmp_Y = struct2table(features.(fieldsFeatures{cntFeature}));

            % Redefine table column names
            for cntSubFeature = 1 : size(fieldsFeatureSub,1)
                tmp_Y.Properties.VariableNames{cntSubFeature} = [nameFeature,'_', tmp_Y.Properties.VariableNames{cntSubFeature}];
            end

            % Append subfeatures to overall Y-variable
            Y = [Y, tmp_Y];

        end

    end
    
    % add observation properties to feature struct
    if exist('featuresProp')
        Y.Label = featuresProp.(opts.useLabel);
    end

end

