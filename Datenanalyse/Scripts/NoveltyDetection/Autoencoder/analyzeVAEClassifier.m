function testData = analyzeVAEClassifier(varargin)

    plot_postprocessing_analsis = 0;

    if find(strcmp(varargin,'origDataPath'))
        origDataPath = varargin{find(strcmp(varargin,'origDataPath'))+1};
    else
        origDataPath = 'E:\Lehrstuhl\Fahrwerkdiagnose\Datenanalyse\BMW\MachineLearning\ManuelleFeatures\200312_0831\origData_woCorruptions.mat';
    end

    if find(strcmp(varargin,'pathResultFolder'))
        pathResultFolder = varargin{find(strcmp(varargin,'pathResultFolder'))+1};
    else
        pathResultFolder = {'test','dataFull','E:\Lehrstuhl\Fahrwerkdiagnose\Datenanalyse\BMW\VAE\2020_03_30_22h_46_57_pvae_normalTrainData\result\test';...
            'val','dataFull','E:\Lehrstuhl\Fahrwerkdiagnose\Datenanalyse\BMW\VAE\2020_03_30_22h_46_57_pvae_normalTrainData\result\val'};
    end

    if find(strcmp(varargin,'origData'))
        origData = varargin{find(strcmp(varargin,'origData'))+1};
    else
        if iscell(pathResultFolder)
            origData = load(origDataPath, pathResultFolder{:,2});
        else
            origData = load(origDataPath, 'dataFull');
            origData = origData.dataFull;
        end
    end
   
    f = [pathResultFolder(:,1)'];
    f{2,1} = {[]};
    testData = struct(f{:});
    for cntPath = 1 : size(pathResultFolder,1)
        tmpPath = pathResultFolder{cntPath,3};
        tmpOrigName = pathResultFolder{cntPath,2};
        tmpName = pathResultFolder{cntPath,1};

        tmpOrigData = origData.(tmpOrigName);
        if ~isfield(tmpOrigData,'Prop')
            tmpOrigData = tmpOrigData.data;
        end
        
        % find number of cvs
        foldersInPath = dir(tmpPath);
        numCV = length(foldersInPath)-2;
        if strcmp(tmpName,'val')
            numCV = 1;
        end
        
        structTest = cell(numCV,1);
        for cntCV = 0 : numCV-1
            structTest{cntCV+1} = importVAEResult([tmpPath,'\cv',num2str(cntCV)], tmpOrigData, '');
        end
        testData.(tmpName) = structTest{1};
        testData.(tmpName).features.Prop.Posterior = cell(numCV,1);
        testData.(tmpName).features.Prop.predictedClass = cell(numCV,1);
        for cntCV = 1 : numCV
            testData.(tmpName).features.Prop.Posterior{cntCV} = structTest{cntCV}.features.Prop.Posterior;
        end
        
        uniqueClasses = testData.(tmpName).features.uniqueClasses;
        if max(strcmp(uniqueClasses,'good'))
            classIntact = 'good';
        elseif max(strcmp(uniqueClasses,'passive intact'))
            classIntact = 'passive intact';
        elseif max(strcmp(uniqueClasses,'passiveIntact'))
            classIntact = 'passiveIntact';
        elseif max(strcmp(uniqueClasses,'allDampersStiff'))
            classIntact = 'allDampersStiff';
        elseif max(strcmp(uniqueClasses,'intact'))
            classIntact = 'intact';
        end
        
        auc = zeros(1,numCV);
        for cntCV = 1 : numCV
            auc(cntCV) = noveltyAUC(testData.(tmpName).features.Prop.Posterior{cntCV}, testData.(tmpName).features.Label, classIntact);
        end
        testData.(tmpName).result.mValue.raw = auc;
        testData.(tmpName).result.mValue.mean = mean(auc);
        testData.(tmpName).result.mValue.std = std(auc);
        
    end
    
    % Evaluate Posterior Probabilities
    fieldsTestData = fields(testData);
    if max(strcmp(fieldsTestData,'val'))

        valNovScore = testData.val.features.Prop.Posterior{1};
        if isfield(testData.val.features.Prop,'batch')
            valBatch = testData.val.features.Prop.batch;
        else
            valBatch = testData.val.features.Prop.ID;
        end
        valObsID = testData.val.features.Prop.observationID;
        
        LDSmodel = trainLDS(valNovScore,valBatch, valObsID);

        fieldsTestset = pathResultFolder(:,1);
        fieldsTestset(strcmp(fieldsTestset,'val')) = [];
        for cntTestset = 1 : length(fieldsTestset)
            
            % find number of cvs
            tmpTestset = fieldsTestset{cntTestset};
            if ~strcmp(tmpTestset,'val')
                foldersInPath = dir(pathResultFolder{strcmp(pathResultFolder,tmpTestset),3});
                numCV = length(foldersInPath)-2;
            else
                numCV = 1;
            end
            
            currTestsetName = fieldsTestset{cntTestset};
            testData.(currTestsetName).probabilityAnalysis = struct();
            
            testData.(currTestsetName).probabilityAnalysis.LDS = struct();
            testData.(currTestsetName).probabilityAnalysis.LDS.Probabilities = cell(numCV,1);
            testData.(currTestsetName).probabilityAnalysis.LDS.ProbabilitiesCov = cell(numCV,1);
            testData.(currTestsetName).probabilityAnalysis.LDS.auc.raw = zeros(numCV,1);
            for cntCV = 1 : numCV
                if isfield(testData.(currTestsetName).features.Prop,'batch')
                    testBatch = testData.(currTestsetName).features.Prop.batch;
                else
                    testBatch = testData.(currTestsetName).features.Prop.ID;
                end
                [testData.(currTestsetName).probabilityAnalysis.LDS.Probabilities{cntCV}, ...
                    testData.(currTestsetName).probabilityAnalysis.LDS.ProbabilitiesCov{cntCV}] = ...
                    calcLDS(testData.(currTestsetName).features.Prop.Posterior{cntCV},...
                    testBatch, testData.(currTestsetName).features.Prop.observationID,...
                    LDSmodel);
                [testData.(currTestsetName).probabilityAnalysis.LDS.auc.raw(cntCV),~,~] = fastAUC(logical(testData.(currTestsetName).features.Prop.LabelID),testData.(currTestsetName).probabilityAnalysis.LDS.Probabilities{cntCV},0);
            end
            testData.(currTestsetName).probabilityAnalysis.LDS.auc.mean = mean(testData.(currTestsetName).probabilityAnalysis.LDS.auc.raw);
            testData.(currTestsetName).probabilityAnalysis.LDS.auc.std = std(testData.(currTestsetName).probabilityAnalysis.LDS.auc.raw);
        end
        
        % Plot Kalman-Filter Visualization
        if plot_postprocessing_analsis
            plot_LDS_Analysis(testData)
        end

    end
end