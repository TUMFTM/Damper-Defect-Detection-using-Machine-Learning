% Script for analysis of the observability of damper defects with varying
% vehicle speed

    
%% Add here data to dataFull
fprintf('Extracting here information...');
tic;
dataFull.Prop.here = addHereDataToProp(dataFull.Prop);
dataFull.Prop.AvgIRI = dataFull.Prop.here.AvgIRI;
dataFull.Prop.RoughnessCat = dataFull.Prop.here.RoughnessCat;
dataFull.Prop = rmfield(dataFull.Prop, 'here');
fprintf('finished\n');
toc

%     figure();
%     [histFull, ~] = histcounts(dataFull.Prop.AvgIRI,iri_vec,'Normalization','count');
%     bar(iri_vec(1:end-1)+diff(iri_vec)/2, histFull);


%% Classify for Speed Analysis
speed_vec = [50:5:100];
for cntSpeed = 1 : length(speed_vec)-1
    if cntSpeed < length(speed_vec)-1
        tmp = reduceDataWithProperty(dataFull,'meanWheelspeed',@(x)((x>=speed_vec(cntSpeed))&(x<speed_vec(cntSpeed+1))));
    else
        tmp = reduceDataWithProperty(dataFull,'meanWheelspeed',@(x)((x>=speed_vec(cntSpeed))&(x<=speed_vec(cntSpeed+1))));
    end
    tmpIdx = cvpartition(tmp.Prop.LabelID,'HoldOut',length(tmp.Prop.observationID)-500,'Stratify',true);
    tmp = removeCorruptedObservations(tmp,tmpIdx.test,'newObsID',0);
    if cntSpeed == 1
        dataTrainingPropAnalysis_Speed = tmp;
    else
        dataTrainingPropAnalysis_Speed = mergeStructs(dataTrainingPropAnalysis_Speed, tmp,1);
    end
end
dataTestingPropAnalysis_Speed = reduceDataWithProperty(dataFull,'observationID',@(x)~ismember(x,dataTrainingPropAnalysis_Speed.Prop.observationID));
featuresTrainingPropAnalysis_Speed = generateFeatureStruct(dataTrainingPropAnalysis_Speed, featureExtractionHandle);
classifierOptsPropAnalysis_Speed = classifierOpts;
classifierOptsPropAnalysis_Speed.optimizeHyperparameters = '';
trainedClassifierPropAnalysis_Speed = trainClassifier(featuresTrainingPropAnalysis_Speed, classifierOptsPropAnalysis_Speed);
testdataTestingPropAnalysis_Speed = testClassifier(trainedClassifierPropAnalysis_Speed, featureExtractionHandle, opts, dataTestingPropAnalysis_Speed);

%% Classify for IRI analysis
iri_vec = [0.5 : 0.25 : 3];
[histTesting, binEdges] = histcounts(dataTesting.Prop.AvgIRI,iri_vec,'Normalization','count');
binCenters_Speed = binEdges(1:end-1)+diff(binEdges)/2;
[histTraining, ~] = histcounts(dataTraining.Prop.AvgIRI,binEdges,'Normalization','count');
figure();
bar(binCenters_Speed, histTraining, 'DisplayName','Training Data')
hold on
bar(binCenters_Speed, histTesting, 'DisplayName','Testing Data')
legend('Location','northeast')
xlabel('IRI');
ylabel('Number of Observations');

for cntIRI = 1 : length(iri_vec)-1
    if cntIRI < length(iri_vec)-1
        tmp = reduceDataWithProperty(dataFull,'AvgIRI',@(x)((x>=iri_vec(cntIRI))&(x<iri_vec(cntIRI+1))));
    else
        tmp = reduceDataWithProperty(dataFull,'AvgIRI',@(x)((x>=iri_vec(cntIRI))&(x<=iri_vec(cntIRI+1))));
    end
%         tmp = reduceDataWithProperty(dataFull,'AvgIRI',@(x)((x<iri_vec(cntIRI+1))&(x>iri_vec(cntIRI))));
    if length(tmp.Prop.observationID)-300>1
        tmpIdx = cvpartition(tmp.Prop.LabelID,'HoldOut',length(tmp.Prop.observationID)-300,'Stratify',true);
        tmp = removeCorruptedObservations(tmp,tmpIdx.test,'newObsID',0);
    end
    if cntIRI == 1
        dataTrainingPropAnalysis_IRI = tmp;
    else
        dataTrainingPropAnalysis_IRI = mergeStructs(dataTrainingPropAnalysis_IRI, tmp,1);
    end
end
dataTestingPropAnalysis_IRI = reduceDataWithProperty(dataFull,'observationID',@(x)~ismember(x,dataTrainingPropAnalysis_IRI.Prop.observationID));
featuresTrainingPropAnalysis_IRI = generateFeatureStruct(dataTrainingPropAnalysis_IRI, featureExtractionHandle);
classifierOptsPropAnalysis_IRI = classifierOpts;
classifierOptsPropAnalysis_IRI.optimizeHyperparameters = '';
trainedClassifierPropAnalysis_IRI = trainClassifier(featuresTrainingPropAnalysis_IRI, classifierOptsPropAnalysis_IRI);
testdataTestingPropAnalysis_IRI = testClassifier(trainedClassifierPropAnalysis_IRI, featureExtractionHandle, opts, dataTestingPropAnalysis_IRI);


%% Analyze Speed Dependency
% AUC variation
auc_over_speed = zeros(length(speed_vec)-1,5);
accuracy_over_speed = zeros(length(speed_vec)-1,5);
for cntSpeed = 1 : length(speed_vec)-1
    if cntSpeed < length(speed_vec)-1
        currIdxOfObs = (testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed >= speed_vec(cntSpeed)) & (testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed < speed_vec(cntSpeed+1));
    else
        currIdxOfObs = (testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed >= speed_vec(cntSpeed)) & (testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed <= speed_vec(cntSpeed+1));
    end
    for cntCV = 1 : 5
        auc_over_speed(cntSpeed,cntCV) = multiClassAUC(testdataTestingPropAnalysis_Speed.features.Prop.Posterior{cntCV}(currIdxOfObs,:),vec2ind(testdataTestingPropAnalysis_Speed.features.labelAsMatrix(currIdxOfObs,:)')');
        [cm,~,~,~] = confusion(testdataTestingPropAnalysis_Speed.features.labelAsMatrix(currIdxOfObs,:)',testdataTestingPropAnalysis_Speed.features.Prop.Posterior{cntCV}(currIdxOfObs,:)');
        accuracy_over_speed(cntSpeed,cntCV) = 1-cm;
    end
end
figure;
plot(speed_vec(1:end-1)+diff(speed_vec)/2,100*mean(auc_over_speed,2),'k','LineWidth',1,'DisplayName','AUC');
hold on
plot(speed_vec(1:end-1)+diff(speed_vec)/2,100*mean(accuracy_over_speed,2),'--k','LineWidth',1,'DisplayName','Accuracy');
legend show;
xlabel('Speed in km/h');
ylabel('Accuracy in % / AUC in %')
title('Accuracy and AUC of testing data over Speed');

% Distribution of vehicle speed in training data for Speed
[histTesting, binEdges] = histcounts(testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed,'Normalization','count');
binCenters_Speed = binEdges(1:end-1)+diff(binEdges)/2;
[histTraining, ~] = histcounts(dataTrainingPropAnalysis_Speed.Prop.meanWheelspeed,binEdges,'Normalization','count');
figure();
h = bar(binCenters_Speed', [histTraining', histTesting'], 'grouped');
set(h(1),'FaceColor',[0 0 0]);
set(h(2),'FaceColor',[0.6 0.6 0.6]);
xlim([min(speed_vec),max(speed_vec)]);
legend({'Training';'Testing'})
xlabel('Speed in km/h');
ylabel('Number of Observations');
title('Observation Distribution for Analysis of Speed Influence on Observability');

% Analysis for Speed
TrueClassPosterior = max(testdataTestingPropAnalysis_Speed.features.Prop.Posterior{1,1} .* testdataTestingPropAnalysis_Speed.features.labelAsMatrix, [], 2);
relevantIdx_Speed = (binCenters_Speed<100)&(binCenters_Speed>50);
[histData_Speed, binCenter_Speed] = hist3([testdataTestingPropAnalysis_Speed.features.Prop.meanWheelspeed, TrueClassPosterior],'Nbins',[length(binCenters_Speed) 5]);
histData_Speed_norm = histData_Speed./sum(histData_Speed,2);
figure();
h = bar(binCenters_Speed(relevantIdx_Speed), 100*histData_Speed_norm(relevantIdx_Speed,:),'stacked');
color = [0 0.3 0.5 0.65 0.8];
for cntH = 1 : length(h)
    set(h(cntH),'FaceColor',color(cntH)*ones(3,1));
    if cntH == length(h)
        set(h(cntH),'DisplayName',[num2str(round(100*(binCenter_Speed{2}(cntH)-mean(diff(binCenter_Speed{2}))/2))),' <= p(TrueClass) <= ', num2str(round(100*(binCenter_Speed{2}(cntH)+mean(diff(binCenter_Speed{2}))/2)))]);
    else
        set(h(cntH),'DisplayName',[num2str(round(100*(binCenter_Speed{2}(cntH)-mean(diff(binCenter_Speed{2}))/2))),' <= p(TrueClass) < ', num2str(round(100*(binCenter_Speed{2}(cntH)+mean(diff(binCenter_Speed{2}))/2)))]);
    end
end
legend(fliplr(h));
xlabel('Vehicle Speed in km/h');
ylabel('Percentage share of estimated true class probability');
title('Percentage share of estimated true class probability for each speed category of test data');

% Correlation analysis for Speed
figure;
bar(100*binCenter_Speed{2},corr(histData_Speed_norm(relevantIdx_Speed,:),binCenters_Speed(relevantIdx_Speed)'),'FaceColor',[0.0 0.0 0.0]);
set(gca,'XTick',[10 30 50 70 90]);
set(gca,'XTickLabels',{'0-20';'20-40';'40-60';'60-80';'80-100'});
xlabel('TrueClass Probability in %');
ylabel('Correlation Coefficient');
title('Correlation of Estimated TrueClass Probability and Vehicle Speed For each Probability Share');

%% Analyze IRI Dependency
% AUC variation for IRI
auc_over_iri = zeros(length(iri_vec)-1,5);
accuracy_over_iri = zeros(length(iri_vec)-1,5);
for cntIRI = 1 : length(iri_vec)-1
    currIdxOfObs = (testdataTestingPropAnalysis_IRI.features.Prop.AvgIRI<iri_vec(cntIRI+1))&(testdataTestingPropAnalysis_IRI.features.Prop.AvgIRI>iri_vec(cntIRI));
    for cntCV = 1 : 5
        auc_over_iri(cntIRI,cntCV) = multiClassAUC(testdataTestingPropAnalysis_IRI.features.Prop.Posterior{cntCV}(currIdxOfObs,:),vec2ind(testdataTestingPropAnalysis_IRI.features.labelAsMatrix(currIdxOfObs,:)')');
        [cm,~,~,~] = confusion(testdataTestingPropAnalysis_IRI.features.labelAsMatrix(currIdxOfObs,:)',testdataTestingPropAnalysis_IRI.features.Prop.Posterior{cntCV}(currIdxOfObs,:)');
        accuracy_over_iri(cntIRI,cntCV) = 1-cm;
    end
end
figure;
plot(iri_vec(1:end-1)+diff(iri_vec)/2,100*mean(auc_over_iri,2),'k','LineWidth',1,'DisplayName','AUC');
hold on
plot(iri_vec(1:end-1)+diff(iri_vec)/2,100*mean(accuracy_over_iri,2),'--k','LineWidth',1,'DisplayName','Accuracy');
legend show;
xlabel('IRI');
ylabel('Accuracy in % / AUC in %')
title('Accuracy and AUC of testing data over IRI');

% Distribution of IRI in training data for IRI
[histTesting, binEdges_IRI] = histcounts(testdataTestingPropAnalysis_IRI.features.Prop.AvgIRI,iri_vec,'Normalization','count');
binCenters_IRI = binEdges_IRI(1:end-1)+diff(binEdges_IRI)/2;
[histTraining, ~] = histcounts(dataTrainingPropAnalysis_IRI.Prop.AvgIRI,iri_vec,'Normalization','count');
figure();
h = bar(binCenters_IRI', [histTraining', histTesting'], 'grouped');
set(h(1),'FaceColor',[0 0 0]);
set(h(2),'FaceColor',[0.6 0.6 0.6]);
xlim([min(iri_vec),max(iri_vec)]);
legend({'Training';'Testing'})
xlabel('IRI');
title('Observation Distribution for Analysis of IRI Influence on Observability');

% Analysis for IRI
TrueClassPosterior = max(testdataTestingPropAnalysis_IRI.features.Prop.Posterior{1,1} .* testdataTestingPropAnalysis_IRI.features.labelAsMatrix, [], 2);
relevantIdx_IRI = (binCenters_IRI<=iri_vec(end))&(binCenters_IRI>=iri_vec(1));
[histData_IRI, binCenter_IRI] = hist3([testdataTestingPropAnalysis_IRI.features.Prop.AvgIRI, TrueClassPosterior],'Edges',{binEdges_IRI,[0:0.2:1]});
if min(histData_IRI(:,end))==0
    histData_IRI(:,end)=[];
    binCenter_IRI{2}(end) = [];
end
histData_IRI_norm = histData_IRI./sum(histData_IRI,2);
figure();
h = bar(binCenters_IRI(relevantIdx_IRI), 100*histData_IRI_norm(relevantIdx_IRI,:),'stacked');
color = [0 0.3 0.5 0.65 0.8];
for cntH = 1 : length(h)
    set(h(cntH),'FaceColor',color(cntH)*ones(3,1));
    if cntH == length(h)
        set(h(cntH),'DisplayName',[num2str(round(100*(binCenter_IRI{2}(cntH)-mean(diff(binCenter_IRI{2}))/2))),' <= p(TrueClass) <= ', num2str(round(100*(binCenter_IRI{2}(cntH)+mean(diff(binCenter_IRI{2}))/2)))]);
    else
        set(h(cntH),'DisplayName',[num2str(round(100*(binCenter_IRI{2}(cntH)-mean(diff(binCenter_IRI{2}))/2))),' <= p(TrueClass) < ', num2str(round(100*(binCenter_IRI{2}(cntH)+mean(diff(binCenter_IRI{2}))/2)))]);
    end
end
legend(fliplr(h));
xlabel('IRI');
ylabel('Percentage share of estimated true class probability');
title('Percentage share of estimated true class probability for each IRI category of test data');

% Correlation analysis for IRI and Speed
corr_Speed = corr(histData_Speed_norm(relevantIdx_Speed,:),binCenters_Speed(relevantIdx_Speed)');
corr_IRI = corr(histData_IRI_norm(relevantIdx_IRI,:),binCenters_IRI(relevantIdx_IRI)');

figure;
h = bar(100*binCenter_IRI{2},[corr_Speed, corr_IRI]);
set(h(1),'FaceColor',[0 0 0]);
set(h(1),'DisplayName','Speed');
set(h(2),'FaceColor',[0.6 0.6 0.6]);
set(h(2),'DisplayName','IRI');
set(gca,'XTick',[10 30 50 70 90]);
set(gca,'XTickLabels',{'0-20';'20-40';'40-60';'60-80';'80-100'});
legend('Location','NorthWest');
xlabel('True Class Probability in %');
ylabel('Correlation Coefficient');
title('Correlation of Estimated TrueClass Probability and IRI');

