
% load('W:\Projekte\Fahrwerkdiagnose\Datenanalyse\BMW\RepresentationLearning\Sparsefilter\200311_1053\Workspace.mat');

lineStyle = {'-';'--';'-.';':'};
color_bw = linspace(0.6,0,4)'*[1 1 1];

analysis_SF = cell(size(mdlSparsefilter));

save_analysis = 1;

%% Calc FFT for comparison
fftHandle = @(x)generateFFTDataAsTable(x,opts, 'windowlength', 128, 'windowstep', 64); % 'windowlength', 128, 'windowstep', 64
fftData = generateFeatureStruct(dataFull, fftHandle);
fftMean = struct();
for cntSignal = 1 : length(opts.fieldsData)
    tmp = mean(fftData.dataAsArray(:,contains(fftData.data.Properties.VariableNames,opts.fieldsData(cntSignal))),1);
    fftMean.(opts.fieldsData{cntSignal}) = interp1(linspace(0,opts.fs/2,length(tmp)),tmp,linspace(0,opts.fs/2,size(mdlSparsefilter{cntSignal}.TransformWeights,1)/2));
end
uniqueClasses = fftData.uniqueClasses;
uniqueClasses = uniqueClasses([4 3 1 2]);   % re-order to have a nice order for plotting in Diss
nClasses = length(uniqueClasses);
fieldsData = opts.fieldsData;
nSignals = length(fieldsData);


%% Analyze Sparsefilter weights
for cntMdl = 1 : length(mdlSparsefilter)
    % Calculate FFT of Sparsefilter weights
    analysis_SF{cntMdl}.fft.value = calcFFT(mdlSparsefilter{1,cntMdl}.TransformWeights);
    df = (opts.fs/2) / size(analysis_SF{cntMdl}.fft.value,1);
    analysis_SF{cntMdl}.fft.freq = linspace(0, opts.fs/2, size(analysis_SF{cntMdl}.fft.value,1));
    [tmpmax, idxmax] = max(analysis_SF{cntMdl}.fft.value,[],1);
    
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
    tmp_table.inverseRank = size(featureBlockNames,1) - (tmp_table.rank+1);
    tmp_table.relImportance = tmp_table.inverseRank/size(featureBlockNames,1);
    tmp_table = sortrows(tmp_table);
    
    % Analyze Sparsefilter weights based on Fisher Score and mean Frequency
    meanFreq = analysis_SF{cntMdl}.fft.freq(idxmax)';
    FFTamplitude_at_meanFreq = tmpmax';
    analysis_SF{cntMdl}.FFTFisherScore.meanFreq = meanFreq;
    idxFisherScore = contains(featuresTraining.FisherScore.Score.Name,opts.fieldsData{cntMdl});
    FisherScore = featuresTraining.FisherScore.Score.FisherScore(idxFisherScore);
    analysis_SF{cntMdl}.FFTFisherScore = table(meanFreq, FFTamplitude_at_meanFreq, FisherScore);
    analysis_SF{cntMdl}.FFTFisherScoreSorted = sortrows(analysis_SF{cntMdl}.FFTFisherScore,'meanFreq');
    
    % Analyze SF weights based on RFE importance and complete FFT
    analysis_SF{cntMdl}.weightValueMultRelRFERank = analysis_SF{cntMdl}.fft.value' .* tmp_table.relImportance .* fftMean.(opts.fieldsData{cntMdl});
    analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank = max(abs(analysis_SF{cntMdl}.weightValueMultRelRFERank),[],1);
%     analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank = sum(abs(analysis_SF{cntMdl}.weightValueMultRelRFERank),1);
%     analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank = analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank;
end

% Generate data for barplot
data_barplot = zeros(length(analysis_SF{1}.fft.freq), length(mdlSparsefilter));
fft_freqs = analysis_SF{1}.fft.freq;
for cntMdl = 1 : length(mdlSparsefilter)
    uniqueFreqs = unique(analysis_SF{cntMdl}.FFTFisherScoreSorted.meanFreq);
    for cntf = 1 : length(uniqueFreqs)
        data_barplot(fft_freqs==uniqueFreqs(cntf),cntMdl) = max(analysis_SF{cntMdl}.FFTFisherScoreSorted.FisherScore(analysis_SF{cntMdl}.FFTFisherScoreSorted.meanFreq==uniqueFreqs(cntf)));
    end
    
    hBarFFT = figure;
%     yyaxis left
%     bar(fft_freqs, data_barplot(:,cntMdl));
%     hold on
% %     plot(linspace(0,50,32),analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank);
%     ylabel('Maximal Fisher Score');
    
    yyaxis left
    plot(linspace(0,50,32),analysis_SF{cntMdl}.maxAbsWeightValueMultRelRFERank, '-k','LineWidth',1,'DisplayName','Importance');
    hold on
    ylabel('Importance');
    xlabel('Frequency in Hz');
    
    yyaxis right
    fft_mean = cell(nClasses,1);
    hold on;
    for cntClass = 1 : nClasses
        fft_red = reduceFeaturesToSpecificClass(fftData,uniqueClasses{cntClass});
        fft_mean{cntClass} = mean(table2array(fft_red.data),1);
        plot(linspace(0,opts.fs/2,length(fft_mean{cntClass})/nSignals),fft_mean{cntClass}(contains(fft_red.featureNames,fieldsData{cntMdl})),'DisplayName',uniqueClasses{cntClass}, 'LineWidth', 1, 'LineStyle', lineStyle{cntClass}, 'Color', color_bw(cntClass,:));
    end
    title(fieldsData{cntMdl}, 'Interpreter','None');
    xlabel('Frequency in Hz');
    legend('Interpreter','None');
    grid on
    if save_analysis
        savefig([ctrl.pathToSave,fieldsData{cntMdl},'_SF_Importance_wFFT.fig']);
        matlab2tikz([ctrl.pathToSave,fieldsData{cntMdl},'_SF_Importance_wFFT.tikz']);
        saveas(hBarFFT, [ctrl.pathToSave,fieldsData{cntMdl},'_SF_Importance_wFFT.emf']);
    end
end


% %% Plot FFT Bars of all Speed-related weights
% hBar = figure;
% bar(fft_freqs, data_barplot(:,contains(opts.fieldsData,'SPEED')));
% legend(opts.fieldsData(contains(opts.fieldsData,'SPEED')),'Interpreter','None');
% xlabel('Frequency in Hz');
% ylabel('Maximal Fisher Score');
% grid on
% if save_analysis
%     savefig([ctrl.pathToSave,'SF_Barplot_Speed.fig']);
%     matlab2tikz([ctrl.pathToSave,'SF_Barplot_Speed.tikz']);
%     saveas(hBar, [ctrl.pathToSave,'SF_Barplot_Speed.emf']);
% end
% 
% %% Plot FFT Bars of all Body-related weights
% hBar = figure;
% bar(fft_freqs, data_barplot(:,~contains(opts.fieldsData,'SPEED')));
% legend(opts.fieldsData(~contains(opts.fieldsData,'SPEED')),'Interpreter','None');
% xlabel('Frequency in Hz');
% ylabel('Maximal Fisher Score');
% grid on
% if save_analysis
%     savefig([ctrl.pathToSave,'SF_Barplot_Body.fig']);
%     matlab2tikz([ctrl.pathToSave,'SF_Barplot_Body.tikz']);
%     saveas(hBar, [ctrl.pathToSave,'SF_Barplot_Body.emf']);
% end


%% Plot some Sparsefilter Weights
% Plot Sparsefilter weights over time
selectSFmdl = 4;
selectWeightIdx = [9;11]; 
hWeights = figure;
hold on;
for cntWeight = 1 : length(selectWeightIdx)
    plot(linspace(0,size(mdlSparsefilter{selectSFmdl}.TransformWeights,1)/opts.fs, size(mdlSparsefilter{selectSFmdl}.TransformWeights,1)),...
        mdlSparsefilter{selectSFmdl}.TransformWeights(:,selectWeightIdx(cntWeight)),'k','LineWidth',1,'LineStyle',lineStyle{cntWeight},'DisplayName',['Kernel ',num2str(selectWeightIdx(cntWeight))]);
end
xlabel('Time in sec');
ylabel('Weight');
title(['SF Weights - ', opts.fieldsData{selectSFmdl}],'Interpreter','None');
legend('Interpreter','None');
grid on
if save_analysis
    savefig([ctrl.pathToSave,'SF_Weights.fig']);
    matlab2tikz([ctrl.pathToSave,'SF_Weights.tikz']);
    saveas(hWeights, [ctrl.pathToSave,'SF_Weights.emf']);
end

% Plot FFT of Sparsefilter weights
figure;
hold on;
for cntWeight = 1 : length(selectWeightIdx)
    plot(analysis_SF{selectSFmdl}.fft.freq, analysis_SF{selectSFmdl}.fft.value(:,selectWeightIdx(cntWeight)),'k','LineWidth',1,'LineStyle',lineStyle{cntWeight},'DisplayName',['Kernel ',num2str(selectWeightIdx(cntWeight))])
end
xlabel('Frequency in Hz');
ylabel('FFT Amplitude');
title(['FFT of SF Weights - ', opts.fieldsData{selectSFmdl}],'Interpreter','None');
legend('Interpreter','None');
grid on
if save_analysis
    savefig([ctrl.pathToSave,'SF_Weights_FFTed.fig']);
    matlab2tikz([ctrl.pathToSave,'SF_Weights_FFTed.tikz']);
    saveas(hWeights, [ctrl.pathToSave,'SF_Weights_FFTed.emf']);
end

