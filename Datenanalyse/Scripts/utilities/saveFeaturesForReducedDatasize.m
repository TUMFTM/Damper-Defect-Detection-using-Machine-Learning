function saveFeaturesForReducedDatasize(datasizeAnalysis, opts, ctrl, dataTrainingFull, dataTesting, varargin)
%SAVEFEATURESFORREDUCEDDATASIZE Summary of this function goes here
%   Detailed explanation goes here

    if find(strcmp(varargin,'testDD2Mass'))
        testDD2Mass = varargin{find(strcmp(varargin,'testDD2Mass'))+1};
    else
        testDD2Mass = 0;
    end
    
    if find(strcmp(varargin,'testDD2Tire'))
        testDD2Tire = varargin{find(strcmp(varargin,'testDD2Tire'))+1};
    else
        testDD2Tire = 0;
    end

    for cntSize = 1 : length(datasizeAnalysis.absSizeFeaturesTraining)
        
        dataTrainingReduced = removeCorruptedObservations(dataTrainingFull, datasizeAnalysis.removedTrainingSamples{cntSize,1});
        featuresTraining = generateFeatureStruct(dataTrainingReduced, datasizeAnalysis.featureExtractionHandle{cntSize,1});
        testdataTesting = testClassifier(datasizeAnalysis.trainedClassifier{cntSize,1}, datasizeAnalysis.featureExtractionHandle{cntSize,1}, opts, dataTesting);
        
        if isstruct(testDD2Mass)
            testDD2Mass = testClassifier(datasizeAnalysis.trainedClassifier{cntSize,1}, datasizeAnalysis.featureExtractionHandle{cntSize,1}, opts, testDD2Mass.data);
        end
        if isstruct(testDD2Tire)
            testDD2Tire = testClassifier(datasizeAnalysis.trainedClassifier{cntSize,1}, datasizeAnalysis.featureExtractionHandle{cntSize,1}, opts, testDD2Tire.data);
        end
        
        % Create Folder
        statusFolder = mkdir([ctrl.pathToSave,'DatasizeAnalysis/']);
        if ~statusFolder
            % Create folder iteratively
            currPath = pwd;
            for pathPart = split([ctrl.pathToSave,'DatasizeAnalysis/'])
                if ~exist(pathPart{1},'dir')
                    mkdir(pathPart{1});
                end
                cd(pathPart{1});
            end
            cd(currPath);
        end
        
        if isstruct(testDD2Mass) && isstruct(testDD2Tire)
            save([ctrl.pathToSave,'DatasizeAnalysis/',num2str(datasizeAnalysis.absSizeFeaturesTraining(cntSize)),'_TrainingSamples.mat'],'featuresTraining','testdataTesting', 'testDD2Mass', 'testDD2Tire');
        else
            save([ctrl.pathToSave,'DatasizeAnalysis/',num2str(datasizeAnalysis.absSizeFeaturesTraining(cntSize)),'_TrainingSamples.mat'],'featuresTraining','testdataTesting');
        end
    end

end

