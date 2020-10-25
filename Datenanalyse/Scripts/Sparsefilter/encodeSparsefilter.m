function featuresTableOut = encodeSparsefilter(data,mdlSparsefilter,opts,sparsefilterOpts)
%ENCODESEGMENTEDAUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

    
    if strcmp(sparsefilterOpts.Type, 'OneIndividualSparsefilterForEachSignal')
        %% One Individual Sparsefilter for each signal
        features = cell(length(opts.fieldsData),1);

        parfor cntSignal = 1 : length(opts.fieldsData)

            % Segment data
            if sparsefilterOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                inputToSparsefilter = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            else
                inputToSparsefilter = data.(opts.fieldsData{cntSignal});
            end

            % Generate Sparsefilter Features
            tmpFeatures = transform(mdlSparsefilter{1,cntSignal},inputToSparsefilter);

            % Calculate mean of segmented features
            if sparsefilterOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*sparsefilterOpts.numSegmentsPerObservation+1:cntObs*sparsefilterOpts.numSegmentsPerObservation,:),1);
                end
                features{cntSignal,1} = meanFeatures;
            else
                features{cntSignal,1} = tmpFeatures;
            end

        end

        featuresTableOut = table;
        for cntSignal = 1 : length(opts.fieldsData)
            featuresTableOut.(opts.fieldsData{cntSignal}) = features{cntSignal,1};
        end
        
       
    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForAllAppendedSignals')
        %% One Single Sparsefilter for all appended signals  
    
        % Segment data
        dataReshaped = data;
        if sparsefilterOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                dataReshaped.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals
        dataReshaped = rmfield(dataReshaped,'Prop');
        inputToSparsefilter = table2array(struct2table(dataReshaped));
        
        % Generate Sparsefilter Features
        tmpFeatures = transform(mdlSparsefilter{1},inputToSparsefilter);
        
        % Calculate mean of segmented features
        featuresTableOut = table;
        if sparsefilterOpts.lengthSegment > 0
            meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
            for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*sparsefilterOpts.numSegmentsPerObservation+1:cntObs*sparsefilterOpts.numSegmentsPerObservation,:),1);
            end
            featuresTableOut.Feature = meanFeatures;
        else
            featuresTableOut.Feature = tmpFeatures;
        end

        

    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForAllSignals')
        %% One Single Sparsefilter For All Signals
        
        features = cell(length(opts.fieldsData),1);

        parfor cntSignal = 1 : length(opts.fieldsData)

            % Segment data
            if sparsefilterOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                inputToSparsefilter = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            else
                inputToSparsefilter = data.(opts.fieldsData{cntSignal});
            end

            % Generate Sparsefilter Features
            tmpFeatures = transform(mdlSparsefilter{1},inputToSparsefilter);

            % Calculate mean of segmented features
            if sparsefilterOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*sparsefilterOpts.numSegmentsPerObservation+1:cntObs*sparsefilterOpts.numSegmentsPerObservation,:),1);
                end
                features{cntSignal,1} = meanFeatures;
            else
                features{cntSignal,1} = tmpFeatures;
            end

        end

        featuresTableOut = table;
        for cntSignal = 1 : length(opts.fieldsData)
            featuresTableOut.(opts.fieldsData{cntSignal}) = features{cntSignal,1};
        end
        
        
    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForWheelSpeedsAndOneIndividualSparsefilterForRest')
        %% OneSingleSparsefilterForWheelSpeedsAndOneIndividualSparsefilterForRest
        
        features = cell(length(opts.fieldsData),1);

        for cntSignal = 1 : length(opts.fieldsData)

            % Segment data
            if sparsefilterOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                inputToSparsefilter = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            else
                inputToSparsefilter = data.(opts.fieldsData{cntSignal});
            end
            
            % Generate Sparsefilter Features
            tmpFeatures = transform(mdlSparsefilter{1,cntSignal},inputToSparsefilter);

            % Calculate mean of segmented features
            if sparsefilterOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*sparsefilterOpts.numSegmentsPerObservation+1:cntObs*sparsefilterOpts.numSegmentsPerObservation,:),1);
                end
                features{cntSignal,1} = meanFeatures;
            else
                features{cntSignal,1} = tmpFeatures;
            end

        end

        featuresTableOut = table;
        for cntSignal = 1 : length(opts.fieldsData)
            featuresTableOut.(opts.fieldsData{cntSignal}) = features{cntSignal,1};
        end

    else
        featuresTableOut = [];
    end
    
end

