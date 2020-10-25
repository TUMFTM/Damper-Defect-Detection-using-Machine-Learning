function [featuresTraining,featuresTesting] = splitFeatureStruct(features,tf, varargin)
%SPLITDATA Summary of this function goes here
%   Detailed explanation goes here

    % Split data to trainings and test data
%     if isfield(features, 'Prop')
%         tmpData = rmfield(features,'Prop');
%     end
%     featuresTraining = opts.data;
%     featuresTesting = opts.data;
%     
%     if isfield(featuresTesting, 'Prop')
%         featuresTraining = rmfield(featuresTraining,'Prop');
%         featuresTesting = rmfield(featuresTesting,'Prop');
%     end
%     
%     for cntFields = 1 : size(opts.fieldsData,1)
%         featuresTraining.(opts.fieldsData{cntFields}) = tmpData.(opts.fieldsData{cntFields})(tf,:);
%         featuresTesting.(opts.fieldsData{cntFields}) = tmpData.(opts.fieldsData{cntFields})(~tf,:);
%     end
% 
%     fieldsProp = fields(features.Prop);
%     for cntFieldsProp = 1 : length(fieldsProp)
%         if size(features.Prop.(fieldsProp{cntFieldsProp}),1) == size(tf,1)
%             featuresTraining.Prop.(fieldsProp{cntFieldsProp}) = features.Prop.(fieldsProp{cntFieldsProp})(tf,:);
%             featuresTesting.Prop.(fieldsProp{cntFieldsProp}) = features.Prop.(fieldsProp{cntFieldsProp})(~tf,:);
%         else
%             featuresTraining.Prop.(fieldsProp{cntFieldsProp}) = features.Prop.(fieldsProp{cntFieldsProp});
%             featuresTesting.Prop.(fieldsProp{cntFieldsProp}) = features.Prop.(fieldsProp{cntFieldsProp});
%         end
%     end
    
    if isempty(tf)
        if find(strcmp(varargin,'splitTrainingData'))
            splitTrainingData = varargin{find(strcmp(varargin,'splitTrainingData'))+1};
        else
            splitTrainingData = 0.7;
        end
        n = size(features.data,1);
        tf = false(n,1);
        if splitTrainingData < 1
            tf(1:round(splitTrainingData*n)) = true;
        else
            tf(1:round(splitTrainingData)) = true;
        end
        tf = tf(randperm(n));
    end

    featuresTraining = features;
    featuresTesting = features;
    fields = fieldnames(features);
    for cntFields = 1:numel(fields)
      aField = fields{cntFields};
      if isstruct(features.(aField))
          afields = fieldnames(features.(aField));
          for cntSubFields = 1:numel(afields)
              bField = afields{cntSubFields};
              if size(features.(aField).(bField),1) == size(tf,1)
                   featuresTraining.(aField).(bField) = features.(aField).(bField)(tf,:);
                   featuresTesting.(aField).(bField) = features.(aField).(bField)(~tf,:);
              else
                  featuresTraining.(aField).(bField) = features.(aField).(bField);
                  featuresTesting.(aField).(bField) = features.(aField).(bField);
              end
          end
          continue
      end
      if size(features.(aField),1) == size(tf,1)
          featuresTraining.(aField) = features.(aField)(tf,:);
          featuresTesting.(aField) = features.(aField)(~tf,:);
      else
          featuresTraining.(aField) = features.(aField);
          featuresTesting.(aField) = features.(aField);
      end
    end

end

