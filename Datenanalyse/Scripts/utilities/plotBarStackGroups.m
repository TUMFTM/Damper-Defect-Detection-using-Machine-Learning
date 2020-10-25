function [h, hfig] = plotBarStackGroups(stackData, groupLabels)
%% Plot a set of stacked bars, but group them according to labels provided.
%
% Params: 
%      stackData is a 3D matrix (i.e., stackData(i, j, k) => (Group, Stack, StackElement)) 
%      groupLabels is a CELL type (i.e., { 'a', 1 , 20, 'because' };)
%
% Copyright 2011 Evan Bollig (bollig at scs DOT fsu ANOTHERDOT edu
%
% 
NumGroupsPerAxis = size(stackData, 1);
NumStacksPerGroup = size(stackData, 2);

% TUM_color_bw = [0 0 0;...
%     88, 88, 90;...
%     156, 157, 159;...
%     217, 218, 219;...
%     255, 255, 255]/255;
TUM_color_bw = linspace(0.5,1,NumStacksPerGroup)'*[1 1 1];
% TUM_color_bw = [1;0.8;0.6;0.4;0.2]*[0, 101, 189]/255;


% Count off the number of bins
groupBins = 1:NumGroupsPerAxis;
MaxGroupWidth = 0.65; % Fraction of 1. If 1, then we have all bars in groups touching
groupOffset = MaxGroupWidth/NumStacksPerGroup;
set(groot,'defaulttextinterpreter','None');
set(groot,'defaultLegendInterpreter','None');
set(groot,'defaultAxesTickLabelInterpreter','None');
hfig = figure();
hold on; 
for i=1:NumStacksPerGroup

    Y = squeeze(stackData(:,i,:));
%     if size(Y,2)==1
%         Y(:,2)=0.1;
%     end
    % Center the bars:
    
    internalPosCount = i - ((NumStacksPerGroup+1) / 2);
    
    % Offset the group draw positions:
    groupDrawPos = (internalPosCount)* groupOffset + groupBins;
    
    h(i,:) = bar(Y, 'stacked', 'FaceColor', TUM_color_bw(i,:));
    set(h(i,:),'BarWidth',groupOffset);
    set(h(i,:),'XData',groupDrawPos);
end
hold off;
set(gca,'XTickMode','manual');
set(gca,'XTick',1:NumGroupsPerAxis);
set(gca,'XTickLabelMode','manual');
set(gca,'XTickLabel',groupLabels);

end 
