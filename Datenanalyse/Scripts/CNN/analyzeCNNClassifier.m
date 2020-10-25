function testData = analyzeCNNClassifier(varargin)

    if find(strcmp(varargin,'origDataPath'))
        origDataPath = varargin{find(strcmp(varargin,'origDataPath'))+1};
    else
        origDataPath = 'E:\Lehrstuhl\Fahrwerkdiagnose\Datenanalyse\BMW\MachineLearning\ManuelleFeatures\200312_0831\Workspace.mat';
    end

    if find(strcmp(varargin,'pathResultFolder'))
        pathResultFolder = varargin{find(strcmp(varargin,'pathResultFolder'))+1};
    else
        pathResultFolder = 'E:\Lehrstuhl\Fahrwerkdiagnose\Datenanalyse\CNN\results\2020_03_25_18h_04_fft\addc_2_e_001_poolingsize_002_cv5';
    end

    if find(strcmp(varargin,'origData'))
        origData = varargin{find(strcmp(varargin,'origData'))+1};
    else
        origData = load(origDataPath, 'dataFull');
        origData = origData.dataFull;
    end

    if iscell(pathResultFolder)
        origData = load(origDataPath, pathResultFolder{:,1});
    end

    structVal = cell(5,1);
    for cntCV = 0 : 4
        structVal{cntCV+1} = importCNNResult([pathResultFolder{1,2},'\fold_',num2str(cntCV)], origData.dataFull, '_val');
    end
    valData = structVal{1};
    valData.features.Prop.Posterior = cell(5,1);
    valData.features.Prop.predictedClass = cell(5,1);

    for cntCV = 1 : 5
        valData.features.Prop.Posterior{cntCV} = structVal{cntCV}.features.Prop.Posterior;
        valData.features.Prop.predictedClass{cntCV} = structVal{cntCV}.features.Prop.predictedClass;
    end
    
    f = [pathResultFolder(:,1)'];
    f{2,1} = {[]};
    testData = struct(f{:});
    for cntPath = 1 : size(pathResultFolder,1)
        tmpPath = pathResultFolder{cntPath,2};
        tmpName = pathResultFolder{cntPath,1};

        tmpOrigData = origData.(tmpName);
        if ~isfield(tmpOrigData,'Prop')
            tmpOrigData = tmpOrigData.data;
        end
        
        structTest = cell(5,1);
        for cntCV = 0 : 4
            structTest{cntCV+1} = importCNNResult([tmpPath,'\fold_',num2str(cntCV)], tmpOrigData, '');
        end
        testData.(tmpName) = structTest{1};
        testData.(tmpName).features.Prop.Posterior = cell(5,1);
        testData.(tmpName).features.Prop.predictedClass = cell(5,1);
        for cntCV = 1 : 5
            testData.(tmpName).features.Prop.Posterior{cntCV} = structTest{cntCV}.features.Prop.Posterior;
            testData.(tmpName).features.Prop.predictedClass{cntCV} = structTest{cntCV}.features.Prop.predictedClass;
        end
        testData.(tmpName).result = classifierValidation(testData.(tmpName).features.Prop.Posterior, testData.(tmpName).features.labelAsMatrix, testData.(tmpName).features.uniqueClasses,'generateConfusionMatrix',0);
        testData.(tmpName).probabilityAnalysis = evaluateProbabilityPostProcessing(testData.(tmpName).features, structVal{1,1}.features, []);
        
        if strcmp(tmpName,'testDD2Mass') || strcmp(tmpName, 'testDD2Tire')
            tmp = evaluateProbabilityPostProcessing(testData.(tmpName).features, testData.(tmpName).features, []);
            testData.(tmpName).probabilityAnalysis.hiddenMarkovWithSpecificEmission = tmp.hiddenMarkov;
        end
        
    end

end