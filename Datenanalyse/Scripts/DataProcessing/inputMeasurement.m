function data = inputMeasurement(Label_Data,opts,varargin)
% Script for loading logfile data
% Output is data-struct

    % set control variable
    if find(strcmp(varargin,'publication'))
        ctrl.publication = varargin{find(strcmp(varargin,'publication'))+1};
    else
        ctrl.publication = 0;
    end

    % Pre- and Post-Fixes to be removed
    removePrePostFix = {'ALIV','CRC','QU','ERR','KCAN'};
    removePrePostFixDataSignals = removePrePostFix;
    if isfield(opts, 'dataSource')
        if strcmp(opts.dataSource,'Flexray')
            removePrePostFixDataSignals = [removePrePostFixDataSignals, 'FCAN'];
        elseif strcmp(opts.dataSource,'CAN')
            removePrePostFixDataSignals = [removePrePostFixDataSignals, 'FlexRay'];
        end
    end

    %% Get number of batches
    numBatches = size(Label_Data,1);
    
    %% Identify signal names to be loaded based on opts.data-entries and
    % Identify signal names to be loaded based on opts.data-entries and
    % sensor names in opts.sensor_names
    sensor_names = opts.sensor_names(isfield(opts.data,opts.sensor_names(:,2)),:);
    starredSignalNames = strcat('*',sensor_names(:,1));
    
    PropDataSignalNames = opts.sensor_names(isfield(opts.propData,opts.sensor_names(:,2)),:);
    starredPropDataSignalNames = strcat('*',PropDataSignalNames(:,1));
    
    if ctrl.publication 
        single_data = cell(numBatches, 1);
    end
    
    
    parfor (cntBatch = 1 : numBatches, (opts.useParallel*opts.numParallelWorkers))

        % load specified variables of log file of cntBatch
        filename = [Label_Data.filePath{cntBatch}, '\', Label_Data.filename{cntBatch}, '.mat'];
        fprintf('Loading file %d of %d... ', cntBatch, numBatches);
        
        %% Load relevant data
        filename = strrep(filename, '\', filesep);
        raw_data = load(filename,starredSignalNames{:});

        % remove too much loaded data and convert variable names to a equal specification
        cnt_varNames1 = 1;
        tmp_varNames = fields(raw_data);
        while cnt_varNames1 <= size(tmp_varNames,1)
            
            if contains(tmp_varNames{cnt_varNames1},removePrePostFixDataSignals)
                raw_data = rmfield(raw_data,tmp_varNames{cnt_varNames1});
                tmp_varNames = fields(raw_data);
            else
                cnt_varNames1 = cnt_varNames1 + 1;
            end
        end
        
        % Copy data to tmp_data
        % initialize tmp_data-struct (needed for parfor)
        tmp_data = opts.data;
        if isfield(tmp_data,'Prop')
            tmp_data = rmfield(tmp_data,'Prop');
        end
        
        for cnt_varNames = 1 : size(sensor_names,1)
            % localize sensor signal in tmp_data
            idx = contains(tmp_varNames,sensor_names{cnt_varNames,1});
            if max(idx)==0
                % signal not found in measurement file
                fprintf('%s not found in %s\n', sensor_names{cnt_varNames,1}, filename);
                continue
            end
            
            % copy sensor signal with correct name to data-struct
            tmp_data.(sensor_names{cnt_varNames,2}) = raw_data.(tmp_varNames{idx});
        end
        
        %% Load additional Prop-data
        if isempty(PropDataSignalNames)
            % No additional information found in log file
            fprintf(' - no additional information found');
            tmpPropData = opts.propData;
            tmpPropData.timeNum = [];
        else
            % load specified variables of log file of cntBatch
            rawPropData = load(filename,starredPropDataSignalNames{:});

            % remove too much loaded data and convert variable names to a equal specification
            cnt_varNames1 = 1;
            tmp_varNames = fields(rawPropData);
            while cnt_varNames1 <= size(tmp_varNames,1)
                if contains(tmp_varNames{cnt_varNames1},removePrePostFix)
                    rawPropData = rmfield(rawPropData,tmp_varNames{cnt_varNames1});
                    tmp_varNames = fields(rawPropData);
                else
                    cnt_varNames1 = cnt_varNames1 + 1;
                end
            end

            % Copy data to tmpPropData
            % initialize tmp_time_data-struct (needed for parfor)
            tmpPropData = opts.propData;

            for cnt_varNames = 1 : size(PropDataSignalNames,1)
                % localize sensor signal in tmp_data
                idx = contains(tmp_varNames,PropDataSignalNames{cnt_varNames,1});
                if max(idx)==0
                    % signal not found in measurement file
                    fprintf('%s not found in %s\n', PropDataSignalNames{cnt_varNames,1}, filename);
                    continue
                end
                % copy sensor signal with correct name to data-struct
                tmpPropData.(PropDataSignalNames{cnt_varNames,2}) = rawPropData.(tmp_varNames{idx});
            end
        end

        % Create timestamp
        if ~isempty(tmpPropData.TIME_SEC)
            tmpTimeData2 = datetime(tmpPropData.TIME_YEAR(:,2),tmpPropData.TIME_MONTH(:,2),tmpPropData.TIME_DAY(:,2),floor(tmpPropData.TIME_HOUR(:,2)),floor(tmpPropData.TIME_MIN(:,2)),floor(tmpPropData.TIME_SEC(:,2)),floor((tmpPropData.TIME_SEC(:,2)-floor(tmpPropData.TIME_SEC(:,2)))*1000));
            tmpTimeData3 = [tmpPropData.TIME_YEAR(:,1), datenum(tmpTimeData2)];
            tmpPropData.timeNum = tmpTimeData3;
        else
            % No time information available
            tmpPropData.timeNum = [];
            fprintf(' - no time information');
        end
        
        % Remove Time fields to save memory
        tmpPropData = rmfield(tmpPropData, {'TIME_YEAR';'TIME_MONTH';'TIME_DAY';'TIME_HOUR';'TIME_MIN';'TIME_SEC'});

        if ~ctrl.publication
            % Resample
            [tmp_data, tDataInterp] = resampleData(tmp_data,opts.fs);

            % Add time as string
            if isfield(tmpPropData,'timeNum') && ~isempty(tmpPropData.timeNum)
                tmpPropData.timeStr = datetime(tmpPropData.timeNum,'ConvertFrom','datenum');
            end

            % Copy zeros to Prop-struct if no data is available (to combine files with and without information
            fieldnamesPropData = fieldnames(tmpPropData)';
            for cntFieldNamesPropData = 1 : length(fieldnamesPropData)
                if isempty(tmpPropData.(fieldnamesPropData{cntFieldNamesPropData}))
                    tmpPropData.(fieldnamesPropData{cntFieldNamesPropData})(:,1) = [min(tDataInterp); max(tDataInterp)];
                    tmpPropData.(fieldnamesPropData{cntFieldNamesPropData})(:,2) = [NaN; NaN];
                end
            end

            % Generate Observations
            single_data(cntBatch) = generateObservations(tmp_data, tmpPropData, tDataInterp, opts,Label_Data(cntBatch,:));
            if isempty(single_data(cntBatch).(opts.fieldsData{1}))
                % no observations found
                fprintf(' - generated oberservations: %d\n', 0);
                continue
            end
            fprintf(' - generated oberservations: %d\n', size(single_data(cntBatch).Prop.batch,1));

        else
            % Generate Data For Publication
            single_data{cntBatch} = tmp_data;
            single_data{cntBatch}.Prop = tmpPropData;

        end
    end

    if ctrl.publication == 0
        % Copy all data to the single data-struct
        data = CatStructFields(single_data(1), single_data(2), 1);
        for cntSingleData = 3 : numBatches
            if isempty(single_data(cntSingleData).(opts.fieldsData{1}))
                continue
            end
            data = CatStructFields(data, single_data(cntSingleData), 1);
        end

        fprintf('Loading data successful\n');
        fprintf('%d observations available\n', size(data.(opts.fieldsData{1}),1))
    else
        data = single_data;
    end
    
end