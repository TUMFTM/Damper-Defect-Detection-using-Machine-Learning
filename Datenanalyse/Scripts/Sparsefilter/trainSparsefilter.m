function [mdlSparsefilter] = trainSparsefilter(data,opts,sparsefilterOpts)
%trainSparseFilterForEachSignal Summary of this function goes here
%   Detailed explanation goes here
    IterationLimit = 5000; % 5000
    if strcmp(sparsefilterOpts.Type, 'OneIndividualSparsefilterForEachSignal')
        %% One Individual Sparsefilter for each signal
        mdlSparsefilter = cell(1,size(opts.fieldsData,1));
        parfor cntSignal = 1 : length(opts.fieldsData)
            % Segment data
            if sparsefilterOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                inputToSparsefilter = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            else
                inputToSparsefilter = data.(opts.fieldsData{cntSignal});
            end

            % Train Sparsefilter
            mdlSparsefilter{1,cntSignal} = sparsefilt(inputToSparsefilter, sparsefilterOpts.numFeatures,...
                'IterationLimit',IterationLimit,'VerbosityLevel',0,'Standardize',true);

        end
        
    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForAllAppendedSignals')
        %% One Single Sparsefilter for all appended signals  
        % Segment data
        if sparsefilterOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals
        data = rmfield(data,'Prop');
        inputToSparsefilter = table2array(struct2table(data));

        % Train Sparsefilter
        mdlSparsefilter = sparsefilt(inputToSparsefilter, sparsefilterOpts.numFeatures,...
            'IterationLimit',IterationLimit,'VerbosityLevel',0,'Standardize',true);
        
    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForAllSignals')
        %% One Single Sparsefilter For All Signals
        % Segment data
        if sparsefilterOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals vertically and shuffle rows
        inputToSparsefilter = [];
        for cntSignal = 1 : length(opts.fieldsData)
            inputToSparsefilter = [inputToSparsefilter; data.(opts.fieldsData{cntSignal})];
        end
        inputToSparsefilter = inputToSparsefilter(randperm(size(inputToSparsefilter,1)),:);
        
        % Train Sparsefilter
        mdlSparsefilter = sparsefilt(inputToSparsefilter, sparsefilterOpts.numFeatures,...
            'IterationLimit',IterationLimit,'VerbosityLevel',0,'Standardize',true);
        
        
    elseif strcmp(sparsefilterOpts.Type, 'OneSingleSparsefilterForWheelSpeedsAndOneIndividualSparsefilterForRest')
        %% OneSingleSparsefilterForWheelSpeedsAndOneIndividualSparsefilterForRest
        % Segment data
        if sparsefilterOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/sparsefilterOpts.lengthSegment)*sparsefilterOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',sparsefilterOpts.lengthSegment,[])';
            end
        end
        
        dataRearranged = rmfield(data,{'SPEED_FL','SPEED_FR','SPEED_RL','SPEED_RR','Prop'});
        dataRearranged.SPEED = [data.SPEED_FL; data.SPEED_FR; data.SPEED_RL; data.SPEED_RR];
        dataRearranged.SPEED = dataRearranged.SPEED(randperm(size(dataRearranged.SPEED,1)),:);      % Shuffle data
        
        % Sort field names to have 'SPEED' as first entry
        fieldsDataRearranged = fields(dataRearranged);
        fieldsDataRearrangedSorted{1} = 'SPEED';
        fieldsDataRearrangedSorted(2:1+sum(~strcmp(fieldsDataRearranged, 'SPEED')),1) = fieldsDataRearranged(~strcmp(fieldsDataRearranged, 'SPEED'));
        
        tmpmdlSparsefilter = cell(1,length(fieldsDataRearrangedSorted));
        parfor cntSignal = 1 : length(fieldsDataRearrangedSorted)
            % Train Sparsefilter
            tmpmdlSparsefilter{1,cntSignal} = sparsefilt(dataRearranged.(fieldsDataRearrangedSorted{cntSignal}), sparsefilterOpts.numFeatures,...
                'IterationLimit',IterationLimit,'VerbosityLevel',0,'Standardize',true);
        end
        
        mdlSparsefilter = cell(1,length(opts.fieldsData));
        for cntSignal = 1 : length(opts.fieldsData)
            if contains(opts.fieldsData{cntSignal},'SPEED')
                mdlSparsefilter{1,cntSignal} = tmpmdlSparsefilter{1,1};
            else
                mdlSparsefilter{1,cntSignal} = tmpmdlSparsefilter{1,strcmp(opts.fieldsData{cntSignal},fieldsDataRearrangedSorted)};
            end
        end
        
    else
        mdlSparsefilter = [];

    end

end

