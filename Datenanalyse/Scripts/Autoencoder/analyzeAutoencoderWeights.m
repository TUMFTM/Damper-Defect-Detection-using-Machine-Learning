
% load('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\BMW\RepresentationLearning\Sparsefilter\200402_1114\Workspace.mat');

save_analysis = 0;

lineStyle = {'-';'--';'-.';':'};
color_bw = linspace(0.6,0,4)'*[1 1 1];

%% Calc FFT for comparison
fftHandle = @(x)generateFFTDataAsTable(x,opts, 'windowlength', 128, 'windowstep', 64); % 'windowlength', 128, 'windowstep', 64
fftData = generateFeatureStruct(dataFull, fftHandle);
% fftMean = struct();
% for cntSignal = 1 : length(opts.fieldsData)
%     tmp = mean(fftData.dataAsArray(:,contains(fftData.data.Properties.VariableNames,opts.fieldsData(cntSignal))),1);
%     fftMean.(opts.fieldsData{cntSignal}) = interp1(linspace(0,opts.fs/2,length(tmp)),tmp,linspace(0,opts.fs/2,size(mdlAutoencoder{cntSignal}.EncoderWeights,2)));
% end
uniqueClasses = fftData.uniqueClasses;
uniqueClasses = uniqueClasses([4 3 1 2]);   % re-order to have a nice order for plotting in Diss
nClasses = length(uniqueClasses);
fieldsData = opts.fieldsData;
nSignals = length(fieldsData);

analysis_AE = cell(size(mdlAutoencoder));

h(1) = figure();
h(2) = figure();
cntStyleSpeed = 0;
cntStyleBody = 0;

for cntMdl = 1 : length(mdlAutoencoder)
    analysis_AE{cntMdl}.weightValue = mdlAutoencoder{cntMdl}.EncoderWeights;
    analysis_AE{cntMdl}.weightFreq = linspace(0,(opts.fs/2),size(mdlAutoencoder{cntMdl}.EncoderWeights,2));
%     analysis_AE{cntMdl}.weightFreq = [(opts.fs/2)/size(mdlAutoencoder{cntMdl}.EncoderWeights,2) : (opts.fs/2)/size(mdlAutoencoder{cntMdl}.EncoderWeights,2) : opts.fs/2];
    
    idxFisherScore = contains(featuresTraining.FisherScore.Score.Name,opts.fieldsData{cntMdl});
    FisherScore = featuresTraining.FisherScore.Score.FisherScore(idxFisherScore);
    
    % transform RFE feature ranking to feature importance
    featureNamesSplitted = split(RFEusingAccuracy.sortedByRank.Name,'_');
    featureBlockNames = join(featureNamesSplitted(:,1:2),'_');
    tmp_rank = find(contains(featureBlockNames,opts.fieldsData{cntMdl}));
    tmp_featureIndex = zeros(length(tmp_rank),1);
    for cntData = 1 : length(tmp_rank)
        tmp_featureIndex(cntData) = str2double(featureNamesSplitted{tmp_rank(cntData),3});
    end
    tmp_table = table;
    tmp_table.featureIndex = tmp_featureIndex;
    tmp_table.rank = tmp_rank;
    tmp_table.inverseRank = size(featureBlockNames,1) - tmp_table.rank;
    tmp_table = sortrows(tmp_table);
    
    analysis_AE{cntMdl}.inverseRelRFERank = (tmp_table.inverseRank-1)/length(featureNamesSplitted);
    analysis_AE{cntMdl}.weightValueMultRelRFERank = analysis_AE{cntMdl}.weightValue .* analysis_AE{cntMdl}.inverseRelRFERank;
    analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank = max(abs(analysis_AE{cntMdl}.weightValueMultRelRFERank),[],1);
%     analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank = sum(abs(analysis_AE{cntMdl}.weightValueMultRelRFERank),1);
%     analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank = analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank .* fftMean.(opts.fieldsData{cntMdl});
    
    analysis_AE{cntMdl}.FisherScore = FisherScore;
    analysis_AE{cntMdl}.weightValueMultFisher = analysis_AE{cntMdl}.weightValue .* analysis_AE{cntMdl}.FisherScore;
    analysis_AE{cntMdl}.maxAbsWeightValueMultFisher = max(abs(analysis_AE{cntMdl}.weightValueMultFisher),[],1);

    if contains(opts.fieldsData{cntMdl},'SPEED')
        figure(h(1));
        cntStyleSpeed = cntStyleSpeed + 1;
        cntStyle = cntStyleSpeed;
    else
        figure(h(2));
        cntStyleBody = cntStyleBody + 1;
        cntStyle = cntStyleBody;
    end
    plot(analysis_AE{cntMdl}.weightFreq, analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank, 'Color', color_bw(cntStyle,:), 'LineStyle', lineStyle{cntStyle}, 'LineWidth', 1, 'DisplayName', opts.fieldsData{cntMdl});
    hold on
    
    % plot individual figure
    hSingle = figure;
    yyaxis left
    plot(analysis_AE{cntMdl}.weightFreq, analysis_AE{cntMdl}.maxAbsWeightValueMultRelRFERank, '-k', 'LineWidth', 1, 'DisplayName', 'AE Weight');
    ylabel('Amplitude');
%     ylim([0 8])
    title('max(abs(InverseRelRFERank*AEWeight)) Autoencoder Weights');
    
    yyaxis right
    fft_mean = cell(nClasses,1);
    hold on;
    for cntClass = 1 : nClasses
        fft_red = reduceFeaturesToSpecificClass(fftData,uniqueClasses{cntClass});
        fft_mean{cntClass} = mean(table2array(fft_red.data),1);
        plot(linspace(0,opts.fs/2,length(fft_mean{cntClass})/nSignals),fft_mean{cntClass}(contains(fft_red.featureNames,fieldsData{cntMdl})),'Color',color_bw(cntClass,:),'LineStyle',lineStyle{cntClass},'DisplayName',['FFT ',uniqueClasses{cntClass}], 'LineWidth', 1);
    end
    title(['max(abs(InverseRelRFERank*AEWeight)) Autoencoder Weights ', fieldsData{cntMdl}], 'Interpreter','None');
    ylabel('FFT Amplitude');
    xlabel('Frequency in Hz');
    legend('Location','best', 'Interpreter','None');
    grid on
    if save_analysis
        savefig([ctrl.pathToSave,fieldsData{cntMdl},'_AE_Weights_wFFT.fig']);
        matlab2tikz([ctrl.pathToSave,fieldsData{cntMdl},'_AE_Weights_wFFT.tikz']);
        saveas(hSingle, [ctrl.pathToSave,fieldsData{cntMdl},'_AE_Weights_wFFT.emf']);
    end

end

for cnth = 1 : length(h)
    figure(h(cnth));
    xlabel('Frequency in Hz');
    ylabel('Amplitude');
    title('max(abs(InverseRelRFERank*AEWeight)) Autoencoder Weights');
    legend('Interpreter', 'none');
    grid on
end

if save_analysis
    savefig(h(1), [ctrl.pathToSave,'AEWeightAnalysis_Speed.fig']);
    matlab2tikz('figurehandle', h(1), [ctrl.pathToSave,'AEWeightAnalysis_Speed.tikz']);
    saveas(h(1), [ctrl.pathToSave,'AEWeightAnalysis_Speed.emf']);
    savefig(h(2), [ctrl.pathToSave,'AEWeightAnalysis_Body.fig']);
    matlab2tikz('figurehandle', h(2), [ctrl.pathToSave,'AEWeightAnalysis_Body.tikz']);
    saveas(h(2), [ctrl.pathToSave,'AEWeightAnalysis_Body.emf']);
end

%% Plot some Sparsefilter Weights
% selectMdl = 7;
selectMdl = 4;
% selectWeightIdx = [11:20]'; %20 31 45
selectWeightIdx = [1]; % [20;45] [1;10]
hWeights = figure;
yyaxis left
hold on
x = linspace(0,opts.fs/2,size(mdlAutoencoder{selectMdl}.EncoderWeights,2));
% x = [(opts.fs/2)/size(mdlAutoencoder{selectMdl}.EncoderWeights,2) : (opts.fs/2)/size(mdlAutoencoder{selectMdl}.EncoderWeights,2) : opts.fs/2];
for cntIdx = 1 : length(selectWeightIdx)
    plot(x,mdlAutoencoder{selectMdl}.EncoderWeights(selectWeightIdx(cntIdx),:),'Color','k','LineStyle',lineStyle{cntIdx},'LineWidth',1, 'DisplayName', ['Kernel ', num2str(selectWeightIdx(cntIdx))]);
end
xlabel('Frequency in Hz');
ylabel('Weight');
% title(['AE Weights - ', opts.fieldsData{selectMdl}],'Interpreter','None');
% legend('Interpreter','None');
ylim([-4 4])
grid on

yyaxis right
fft_mean = cell(nClasses,1);
hold on;
for cntClass = 1 : nClasses
    fft_red = reduceFeaturesToSpecificClass(fftData,uniqueClasses{cntClass});
    fft_mean{cntClass} = mean(table2array(fft_red.data),1);
    plot(linspace(0,opts.fs/2,length(fft_mean{cntClass})/nSignals),fft_mean{cntClass}(contains(fft_red.featureNames,fieldsData{cntMdl})),'Color',color_bw(cntClass,:),'LineStyle',lineStyle{cntClass},'DisplayName',['FFT ',uniqueClasses{cntClass}], 'LineWidth', 1);
end
title(['AE Weights - ', opts.fieldsData{selectMdl}],'Interpreter','None');
ylabel('FFT Amplitude');
xlabel('Frequency in Hz');
ylim([-0.035 0.035])
legend('Interpreter','None');

if save_analysis
    savefig([ctrl.pathToSave,'AEWeights.fig']);
    matlab2tikz([ctrl.pathToSave,'AEWeights.tikz']);
    saveas(hWeights, [ctrl.pathToSave,'AEWeights.emf']);
end

