function output = testClassifier(classifier, funHandleFeatureExtraction, opts, varargin)
%TESTCLASSIFIER Summary of this function goes here
%   Example1: output = testClassifier(classifier, featureExtractionHandle, opts, filename, foldername)
%   Example2: output = testClassifier(classifier, featureExtractionHandle, opts, data)
%   Example3: output = testClassifier(classifier, featureExtractionHandle, opts, features)

    % a dummy variable for 'classifier' is needed, because it is not known
    % if the classifier variable is cell or a classifier
    plotHistogramAnalysis = 1;
    
    if iscell(classifier)
        classifierDummy = classifier{1,1};
    else
        classifierDummy = classifier;
    end

    if find(strcmp(varargin,'addHereDataToProp'))
        ctrl.addHereDataToProp = varargin{find(strcmp(varargin,'addHereDataToProp'))+1};
    else
        ctrl.addHereDataToProp = 0;
    end
    
    if find(strcmp(varargin,'figureName'))
        idxVararginFigureName = find(strcmp(varargin,'figureName'));
        ctrl.figureName = varargin{idxVararginFigureName+1};
        varargin(idxVararginFigureName:idxVararginFigureName+1) = [];
    else
        ctrl.figureName = '';
    end

    if length(varargin)==2
        
        filename = varargin{1};
        foldername = varargin{2};
        data = loadData(filename, foldername, opts);
        
        % Some further chose of data
        if isfield(opts, 'useFFT') && opts.useFFT==1
            data = generateFFTData(data, opts);
        end
        if isfield(opts, 'reduceDataWithProperty')
            data = reduceDataWithProperty(data,opts.reduceDataWithProperty.propertyName,opts.reduceDataWithProperty.functionHandle);
        end
        
        if ~isempty(classifierDummy)
            ClassNames = classifierDummy.ClassNames;
        else
            ClassNames = [];
        end
        features = generateFeatureStruct(data, funHandleFeatureExtraction, 'uniqueClasses', ClassNames);
        
    elseif length(varargin)==1
        
        if isfield(varargin{1}, 'data')
            % Input is feature-struct
            features = varargin{1};
            
            % Some further chose of data
            if isfield(opts, 'reduceDataWithProperty')
                features = reduceDataWithProperty(features,opts.reduceDataWithProperty.propertyName,opts.reduceDataWithProperty.functionHandle);
            end
        else
            % Input is data-struct
            data = varargin{1};
            
            % Some further chose of data
            if isfield(opts, 'useFFT') && opts.useFFT==1
                if (~isfield(data.Prop, 'isFFT')) || (isfield(data.Prop, 'isFFT') && data.Prop.isFFT==0)
                    fprintf('FFT wird berechnet\n');
                    data = generateFFTData(data, opts);
                end
            end
            if isfield(opts, 'reduceDataWithProperty')
                data = reduceDataWithProperty(data,opts.reduceDataWithProperty.propertyName,opts.reduceDataWithProperty.functionHandle);
            end

            if ~isempty(classifierDummy)
                features = generateFeatureStruct(data, funHandleFeatureExtraction, 'uniqueClasses', classifierDummy.ClassNames);
            else
                features = generateFeatureStruct(data, funHandleFeatureExtraction);
            end
            
        end
        
    else
        fprintf('Wrong input format in testClassifier.m\n');
        fprintf('Ending test procedure\n');
        output = NaN;
        return
    end
    
    if ~isempty(classifierDummy) && ~isfield(features.Prop,'Posterior')
        fprintf('Start prediction...');
        tic;
        [predictedClass, Posterior] = predictClassifier(classifier, features.data);
        fprintf('finished - ');
        toc
        
        fprintf('Performance on test data\n');
        
        % Adapt labelAsMatrix to classes of used classifier
        if ~isequal(classifierDummy.ClassNames, features.uniqueClasses)
            labelAsMatrix = generateLabelAsMatrix(features, 'uniqueClasses', classifierDummy.ClassNames);
        else
            labelAsMatrix = features.labelAsMatrix;
        end
%         result = classifierValidation(Posterior, features.labelAsMatrix, features.uniqueClasses);
            result = classifierValidation(Posterior, labelAsMatrix, classifierDummy.ClassNames);
    else
        if isfield(features.Prop,'Posterior')
            fprintf('Using existing Probabilities\n');
            predictedClass = features.Prop.predictedClass;
            Posterior = features.Prop.Posterior;
            labelAsMatrix = features.labelAsMatrix;
            
            result = classifierValidation(Posterior, labelAsMatrix, classifierDummy.ClassNames);
        else
            predictedClass = cell(size(features.Label));
            Posterior = zeros(size(features.labelAsMatrix));
        end
    end
    
    if exist('data','var')
        data.Prop.Posterior = Posterior;
        data.Prop.predictedClass = predictedClass;
        if ctrl.addHereDataToProp
            fprintf('Extracting here information...');
            tic;
            data.Prop.here = addHereDataToProp(data.Prop);
            fprintf('finished - ');
            toc
        end
        features.Prop = data.Prop;
    else
        features.Prop.Posterior = Posterior;
        features.Prop.predictedClass = predictedClass;
        
        if ~isfield(features.Prop, 'here')
            % Save time if here-field already exists
            if ctrl.addHereDataToProp
                fprintf('Extracting here information...');
                tic;
                features.Prop.here = addHereDataToProp(features.Prop);
                fprintf('finished - ');
                toc
            end
        end
    end

    if plotHistogramAnalysis
        if iscell(features.Prop.Posterior)
            singlePosterior = features.Prop.Posterior{1,1};
            singlePredictedClass = features.Prop.predictedClass{1,1};
        else
            singlePosterior = features.Prop.Posterior;
            singlePredictedClass = features.Prop.predictedClass;
        end
        
        if max(max(singlePosterior))>0

            figure('Name',ctrl.figureName);
            if isfield(features.Prop, 'here')
                histogramAnalysisCatVar(singlePosterior, features.labelAsMatrix, features.Prop.here.RoughnessCat, 'RoughnessCat','handleAxes',subplot(2,4,1));
            end
            if isfield(features.Prop, 'track')
                histogramAnalysisCatVar(singlePosterior, features.labelAsMatrix, features.Prop.track, 'Track','handleAxes',subplot(2,4,2));
            end
            histogramAnalysisCatVar(singlePosterior, features.labelAsMatrix, features.Prop.labelIsolation, 'True Class','handleAxes',subplot(2,4,3));
            histogramAnalysisCatVar(singlePosterior, features.labelAsMatrix, singlePredictedClass, 'Predicted Class','handleAxes',subplot(2,4,4));
            if isfield(features.Prop, 'here')
                histogramAnalysisContVar(singlePosterior, features.labelAsMatrix, features.Prop.here.AvgIRI, 'IRI','handleAxes',subplot(2,4,5));
            elseif isfield(features.Prop, 'aussentemperatur')
                histogramAnalysisContVar(singlePosterior, features.labelAsMatrix, features.Prop.aussentemperatur, 'Aussentemp','handleAxes',subplot(2,4,5));
            end
            histogramAnalysisContVar(singlePosterior, features.labelAsMatrix, features.Prop.meanAx, 'a_x','handleAxes',subplot(2,4,6));
            histogramAnalysisContVar(singlePosterior, features.labelAsMatrix, features.Prop.meanAy, 'a_y','handleAxes',subplot(2,4,7));
            histogramAnalysisContVar(singlePosterior, features.labelAsMatrix, features.Prop.meanWheelspeed, 'Wheelspeed','handleAxes',subplot(2,4,8));
        end
    end

    output.classifier = classifier;
    if exist('data', 'var')
        output.data = data;
    end
    output.features = features;
    if exist('result', 'var')
        output.result = result;
    end
    output.opts = opts;
    if exist('filename','var')
        output.filename = filename;
        output.foldername = foldername;
    end
    
end

