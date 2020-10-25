% Validation of classification model (DETECTION & ISOLATION)
% Run "DataProcessing" and "classificationLearner" first
% Export classification model to workspace as "trainedModel"

%% CTRL confidence level
ctrl_conf_level = 1;
conf_level_min = 0.5;

%% Classifier names
classifier = eval('trainedModel1');

%% Model validation
clear predicted_class predicted_class_detection

ModelValidation_DATA_classification = DATA_classification_singFeat_red;
ModelValidation_dataTesting = dataTesting_singFeat_red;

[m,n] = size(ModelValidation_DATA_classification);
true_class = table2array(ModelValidation_DATA_classification(:,end));
true_class_detection = table2array(ModelValidation_DATA_classification(:,end));
true_class_detection = strrep(true_class_detection,'all','fault');
true_class_detection = strrep(true_class_detection,'FL','fault');
true_class_detection = strrep(true_class_detection,'FR','fault');
true_class_detection = strrep(true_class_detection,'RL','fault');
true_class_detection = strrep(true_class_detection,'RR','fault');

predicted_class = classifier.predictFcn(ModelValidation_DATA_classification);
predicted_class_detection = predicted_class;
predicted_class_detection = strrep(predicted_class_detection,'all','fault');
predicted_class_detection = strrep(predicted_class_detection,'FL','fault');
predicted_class_detection = strrep(predicted_class_detection,'FR','fault');
predicted_class_detection = strrep(predicted_class_detection,'RL','fault');
predicted_class_detection = strrep(predicted_class_detection,'RR','fault');

if isfield(classifier,'ClassificationSVM') == 1 && ctrl_conf_level == 1
    [~,confidence_level] = predict(classifier.ClassificationSVM,table2array(ModelValidation_DATA_classification(:,1:n-1)));
    confidence_level = abs(confidence_level(:,1));
end

for i=1:m
    validation_all_data(i,1) = double(~strcmp(true_class(i),predicted_class(i)));
    validation_all_data_detection(i,1) = double(~strcmp(true_class_detection(i),predicted_class_detection(i)));
end

output(1,1) = m;
output(2,1) = sum(validation_all_data);
output(3,1) = 100*(1-(sum(validation_all_data)/m));
output(4,1) = sum(validation_all_data_detection);
output(5,1) = 100*(1-(sum(validation_all_data_detection)/m));

%% Model validation - test data
clear predicted_class predicted_class_detection
[m,n] = size(ModelValidation_dataTesting);
true_class = table2array(ModelValidation_dataTesting(:,end));
true_class_detection = table2array(ModelValidation_dataTesting(:,end));
true_class_detection = strrep(true_class_detection,'all','fault');
true_class_detection = strrep(true_class_detection,'FL','fault');
true_class_detection = strrep(true_class_detection,'FR','fault');
true_class_detection = strrep(true_class_detection,'RL','fault');
true_class_detection = strrep(true_class_detection,'RR','fault');
predicted_class = classifier.predictFcn(ModelValidation_dataTesting);
predicted_class_detection = predicted_class;
predicted_class_detection = strrep(predicted_class_detection,'all','fault');
predicted_class_detection = strrep(predicted_class_detection,'FL','fault');
predicted_class_detection = strrep(predicted_class_detection,'FR','fault');
predicted_class_detection = strrep(predicted_class_detection,'RL','fault');
predicted_class_detection = strrep(predicted_class_detection,'RR','fault');

for i=1:m
    validation_test_data(i,1) = double(~strcmp(true_class(i),predicted_class(i)));
    validation_test_data_detection(i,1) = double(~strcmp(true_class_detection(i),...
        predicted_class_detection(i)));
end

output(1,2) = m;
output(2,2) = sum(validation_test_data);
output(3,2) = 100*(1-(sum(validation_test_data)/m));
output(4,2) = sum(validation_test_data_detection);
output(5,2) = 100*(1-(sum(validation_test_data_detection)/m));

%% Model validation - all data based on confidence level
if isfield(classifier,'ClassificationSVM') == 1 && ctrl_conf_level ==1
    clear predicted_class predicted_class_detection
    idx =  confidence_level <= conf_level_min;
    [m,n] = size(ModelValidation_DATA_classification(~idx,:));
    true_class = table2array(ModelValidation_DATA_classification(~idx,end));
    true_class_detection = table2array(ModelValidation_DATA_classification(~idx,end));
    true_class_detection = strrep(true_class_detection,'all','fault');
    true_class_detection = strrep(true_class_detection,'FL','fault');
    true_class_detection = strrep(true_class_detection,'FR','fault');
    true_class_detection = strrep(true_class_detection,'RL','fault');
    true_class_detection = strrep(true_class_detection,'RR','fault');
    
    predicted_class = classifier.predictFcn(ModelValidation_DATA_classification(~idx,:));
    predicted_class_detection = predicted_class;
    predicted_class_detection = strrep(predicted_class_detection,'all','fault');
    predicted_class_detection = strrep(predicted_class_detection,'FL','fault');
    predicted_class_detection = strrep(predicted_class_detection,'FR','fault');
    predicted_class_detection = strrep(predicted_class_detection,'RL','fault');
    predicted_class_detection = strrep(predicted_class_detection,'RR','fault');
    
    for i=1:m
        validation_confidence_level(i,1) = double(~strcmp(true_class(i),predicted_class(i)));
        validation_confidence_level_detection(i,1) = double(~strcmp(true_class_detection(i),...
            predicted_class_detection(i)));
    end
    
    output(1,3) = m;
    output(2,3) = sum(validation_confidence_level);
    output(3,3) = 100*(1-(sum(validation_confidence_level)/m));
    output(4,3) = sum(validation_confidence_level_detection);
    output(5,3) = 100*(1-(sum(validation_confidence_level_detection)/m));
    loss(1,1) = output(1,1)-output(1,3);
    loss(1,2) = opts.t*loss(1,1);
end

%% Text Output
fprintf('All data: %.0f observations. %.0f prediction fault(s). Accuracy: %.2f %%.\n',...
    output(1,1),output(2,1),output(3,1));
fprintf('Test data: %.0f observations. %.0f prediction fault(s). Accuracy: %.2f %%.\n',...
    output(1,2),output(2,2),output(3,2));
fprintf('All data (detection): %.0f observations. %.0f prediction fault(s). Accuracy: %.2f %%.\n',...
    output(1,1),output(4,1),output(5,1));
fprintf('Test data (detection): %.0f observations. %.0f prediction fault(s). Accuracy: %.2f %%.\n',...
    output(1,2),output(4,2),output(5,2));
if isfield(classifier,'ClassificationSVM') == 1 && ctrl_conf_level ==1
    fprintf('Confidence level dependent classification: %.0f observations. %.0f prediction fault(s).\n',...
        output(1,3),output(2,3));
    fprintf('Accuracy: %.2f %% (%.2f %% detection). %.0f lost observations (%.0f s).\n',...
        output(3,3),output(5,3),loss(1,1),loss(1,2));
end
