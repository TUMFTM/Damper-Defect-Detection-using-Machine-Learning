function featuresTableOut = encodeAutoencoder(data,mdlAutoencoder,opts,autoencoderOpts)
%ENCODESEGMENTEDAUTOENCODER Summary of this function goes here
%   Detailed explanation goes here

    if strcmp(autoencoderOpts.AutoencoderType, 'OneIndividualAutoencoderForEachSignal')
        %% One Individual Autoencoder for each signal
        features = cell(length(opts.fieldsData),1);

        parfor cntSignal = 1 : length(opts.fieldsData)
            % Segment data
            if autoencoderOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                inputToAutoencoder = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            else
                inputToAutoencoder = data.(opts.fieldsData{cntSignal});
            end

            if isa(mdlAutoencoder{cntSignal},'Autoencoder')
                tmpFeatures = encode(mdlAutoencoder{cntSignal},inputToAutoencoder')';
            elseif isa(mdlAutoencoder{cntSignal},'network')
                tmpFeatures = sim(mdlAutoencoder{cntSignal},inputToAutoencoder')';
            else
                fprintf('Undefined type of object mdlAutoencoder \n');
            end

            % Calculate mean of segmented features
            if autoencoderOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*autoencoderOpts.numSegmentsPerObservation+1:cntObs*autoencoderOpts.numSegmentsPerObservation,:),1);
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
        
       
    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForAllAppendedSignals')
        %% One Single Autoencoder for all appended signals  
    
        % Segment data
        dataReshaped = data;
        if autoencoderOpts.lengthSegment > 0
            for cntSignal = 1 : length(opts.fieldsData)
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                dataReshaped.(opts.fieldsData{cntSignal}) = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            end
        end
        
        % Append all signals
        dataReshaped = rmfield(dataReshaped,'Prop');
        inputToAutoencoder = table2array(struct2table(dataReshaped));
        
        try
            if isa(mdlAutoencoder{1},'Autoencoder')
                tmpFeatures = encode(mdlAutoencoder{1},inputToAutoencoder')';
            elseif isa(mdlAutoencoder{1},'network')
                tmpFeatures = sim(mdlAutoencoder{1},inputToAutoencoder')';
            else
    %             tmpFeatures = [];
                fprintf('Undefined type of object mdlAutoencoder \n');
            end
        catch
            if isa(mdlAutoencoder,'Autoencoder')
                tmpFeatures = encode(mdlAutoencoder,inputToAutoencoder')';
            elseif isa(mdlAutoencoder,'network')
                tmpFeatures = sim(mdlAutoencoder,inputToAutoencoder')';
            else
                fprintf('Undefined type of object mdlAutoencoder \n');
            end
        end
        
        % Calculate mean of segmented features
        featuresTableOut = table;
        if autoencoderOpts.lengthSegment > 0
            meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
            for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*autoencoderOpts.numSegmentsPerObservation+1:cntObs*autoencoderOpts.numSegmentsPerObservation,:),1);
            end
            featuresTableOut.Feature = meanFeatures;
        else
            featuresTableOut.Feature = tmpFeatures;
        end

    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForAllSignals')
        %% One Single Autoencoder For All Signals
        
        features = cell(length(opts.fieldsData),1);

        parfor cntSignal = 1 : length(opts.fieldsData)

            % Segment data
            if autoencoderOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                inputToAutoencoder = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            else
                inputToAutoencoder = data.(opts.fieldsData{cntSignal});
            end

            if isfield(autoencoderOpts, 'differenceAsFeatures') && autoencoderOpts.differenceAsFeatures
                tmpFeatures = predict(mdlAutoencoder{1},inputToAutoencoder')';
                tmpFeatures = abs(tmpFeatures - inputToAutoencoder);
            else
                if isa(mdlAutoencoder{1},'Autoencoder')
                    tmpFeatures = encode(mdlAutoencoder{1},inputToAutoencoder')';
                elseif isa(mdlAutoencoder{1},'network')
                    tmpFeatures = sim(mdlAutoencoder{1},inputToAutoencoder')';
                else
                    fprintf('Undefined type of object mdlAutoencoder \n');
                end
            end

            % Calculate mean of segmented features
            if autoencoderOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*autoencoderOpts.numSegmentsPerObservation+1:cntObs*autoencoderOpts.numSegmentsPerObservation,:),1);
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
        
        
    elseif strcmp(autoencoderOpts.AutoencoderType, 'OneSingleAutoencoderForWheelSpeedsAndOneIndividualAutoencoderForRest')
        %% OneSingleAutoencoderForWheelSpeedsAndOneIndividualAutoencoderForRest
        
        features = cell(length(opts.fieldsData),1);
        for cntSignal = 1 : length(opts.fieldsData)

            % Segment data
            if autoencoderOpts.lengthSegment > 0
                samplesForReshape = floor(size(data.(opts.fieldsData{cntSignal}),2)/autoencoderOpts.lengthSegment)*autoencoderOpts.lengthSegment;
                inputToAutoencoder = reshape(data.(opts.fieldsData{cntSignal})(:,1:samplesForReshape)',autoencoderOpts.lengthSegment,[])';
            else
                inputToAutoencoder = data.(opts.fieldsData{cntSignal});
            end
            
            if isa(mdlAutoencoder{cntSignal},'Autoencoder')
                tmpFeatures = encode(mdlAutoencoder{cntSignal},inputToAutoencoder')';
            elseif isa(mdlAutoencoder{cntSignal},'network')
                tmpFeatures = sim(mdlAutoencoder{cntSignal},inputToAutoencoder')';
            else
                fprintf('Undefined type of object mdlAutoencoder \n');
            end

            % Calculate mean of segmented features
            if autoencoderOpts.lengthSegment > 0
                meanFeatures = zeros(size(data.(opts.fieldsData{1}),1),size(tmpFeatures,2));
                for cntObs = 1 : size(data.(opts.fieldsData{1}),1)
                    meanFeatures(cntObs,:) = mean(tmpFeatures((cntObs-1)*autoencoderOpts.numSegmentsPerObservation+1:cntObs*autoencoderOpts.numSegmentsPerObservation,:),1);
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

