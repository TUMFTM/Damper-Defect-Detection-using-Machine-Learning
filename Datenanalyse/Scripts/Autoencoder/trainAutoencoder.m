function [mdlAutoencoder] = trainAutoencoder(data,opts,autoencoderOpts,varargin)
%TRAINAUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

    
    if find(strcmp(varargin,'optimizeHyperparameter'))
        autoencoderOpts.optimizeHyperparameter = varargin{find(strcmp(varargin,'optimizeHyperparameter'))+1};
    else
        autoencoderOpts.optimizeHyperparameter = 1;
    end

    %% One Individual Autoencoder for each signal
    if strcmp(autoencoderOpts.AutoencoderType, 'OneIndividualAutoencoderForEachSignal')
        for cntSignal = 1 : length(opts.fieldsData) % hier kann auch ein parfor hin
            % Segment data
            if autoencoderOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                inputToAutoencoder = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            else
                inputToAutoencoder = data.(opts.fieldsData{cntSignal});
            end

            % Train Autoencoder
            mdlAutoencoder{1,cntSignal} = ...
                trainStackedAutoencoder(inputToAutoencoder,autoencoderOpts.numFeatures,'optimizeHyperparameter',autoencoderOpts.optimizeHyperparameter);
        end
        
    %% One Single Autoencoder for all appended signals  
    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForAllAppendedSignals')
        % Segment data
        if autoencoderOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals
        dataProp = data.Prop;
        data = rmfield(data,'Prop');
        inputToAutoencoder = table2array(struct2table(data));

        % Train Autoencoder
        mdlAutoencoder = ...
            trainStackedAutoencoder(inputToAutoencoder,autoencoderOpts.numFeatures,'optimizeHyperparameter',autoencoderOpts.optimizeHyperparameter);

    %% One Single Autoencoder For All Signals
    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForAllSignals')
        % Segment data
        if autoencoderOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals vertically and shuffle rows
        inputToAutoencoder = [];
        for cntSignal = 1 : length(opts.fieldsData)
            inputToAutoencoder = [inputToAutoencoder; data.(opts.fieldsData{cntSignal})];
        end
        inputToAutoencoder = inputToAutoencoder(randperm(size(inputToAutoencoder,1)),:);
        
        % Train Autoencoder
        mdlAutoencoder = ...
            trainStackedAutoencoder(inputToAutoencoder,autoencoderOpts.numFeatures,'optimizeHyperparameter',autoencoderOpts.optimizeHyperparameter);
        
    %% OneSingleAutoencoderForWheelSpeedsAndOneIndividualAutoencoderForRest
    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForWheelSpeedsAndOneIndividualAutoencoderForRest')
        % Segment data
        if autoencoderOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                data.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            end
        end
        
        dataRearranged = rmfield(data,{'SPEED_FL','SPEED_FR','SPEED_RL','SPEED_RR','Prop'});
        dataRearranged.SPEED = [data.SPEED_FL; data.SPEED_FR; data.SPEED_RL; data.SPEED_RR];
        dataRearranged.SPEED = dataRearranged.SPEED(randperm(size(dataRearranged.SPEED,1)),:);      % Shuffle data
        
        % Sort field names to have 'SPEED' as first entry
        fieldsDataRearranged = fields(dataRearranged);
        fieldsDataRearrangedSorted{1} = 'SPEED';
        fieldsDataRearrangedSorted(2:1+sum(~strcmp(fieldsDataRearranged, 'SPEED')),1) = fieldsDataRearranged(~strcmp(fieldsDataRearranged, 'SPEED'));
        
        tmpmdlAutoencoder = cell(1,length(fieldsDataRearrangedSorted));
        for cntSignal = 1 : length(fieldsDataRearrangedSorted)
            % Train Autoencoder
            tmpmdlAutoencoder{1,cntSignal} = ...
                trainStackedAutoencoder(dataRearranged.(fieldsDataRearrangedSorted{cntSignal}),autoencoderOpts.numFeatures,'optimizeHyperparameter',autoencoderOpts.optimizeHyperparameter);
        end
        
        mdlAutoencoder = cell(1,length(opts.fieldsData));
        for cntSignal = 1 : length(opts.fieldsData)
            if contains(opts.fieldsData{cntSignal},'SPEED')
                mdlAutoencoder{1,cntSignal} = tmpmdlAutoencoder{1,1};
            else
                mdlAutoencoder{1,cntSignal} = tmpmdlAutoencoder{1,strcmp(opts.fieldsData{cntSignal},fieldsDataRearrangedSorted)};
            end
        end
        
    else
        mdlAutoencoder = [];

    end

end

