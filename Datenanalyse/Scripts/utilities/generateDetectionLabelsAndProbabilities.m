% Achtung:
% Code ist noch nicht auf volle Einsatzfähigkeit geprüft.

testdataTesting.features.Prop.ScoresDetection = [max(testdataTesting.features.Prop.Posterior(:,1:3),[],2), testdataTesting.features.Prop.Posterior(:,4)]./sum([max(testdataTesting.features.Prop.Posterior(:,1:3),[],2), testdataTesting.features.Prop.Posterior(:,4)],2);
testdataTesting.features.ClassLabelsDetection = [max(testdataTesting.features.labelAsMatrix(:,1:3),[],2), testdataTesting.features.labelAsMatrix(:,4)] * [1; 2];
multiClassAUC(testdataTesting.features.Prop.ScoresDetection, testdataTesting.features.ClassLabelsDetection)
classifierValidation(testdataTesting.features.Prop.ScoresDetection, testdataTesting.features.ClassLabelsDetectionAsMatrix, {'defect';'passive'},'generateConfusionMatrix',1);
