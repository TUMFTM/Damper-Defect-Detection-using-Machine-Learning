function [data] = generateObservations(inputData, dataProp, tDataInterp, opts, Label_Data)
%CLEANUPDATA Summary of this function goes here
%   Detailed explanation goes here
    opts.window_size = opts.t*opts.fs;
    if opts.overlap >= opts.window_size
        error('Specified overlap greater or equal than window size');
    end
    IDX = [];
    ENABLE = [];

    fieldsInputData = fields(inputData);

    % Create empty data variable
    data = opts.data;
    
    % Copy data for checking accelerations and speed
    TMP_SPEED = [inputData.SPEED_FL inputData.SPEED_FR inputData.SPEED_RL inputData.SPEED_RR];
    TMP_ACC_X = inputData.ACC_X;
    TMP_ACC_Y = inputData.ACC_Y;

    a = 1;
    nSamples = length(inputData.(fieldsInputData{1}));
    if nSamples < opts.window_size
        idx(1:nSamples) = 0;
    else
        while a+opts.window_size-1 <= nSamples
            b = a+opts.window_size-1;
            enable = [mean(reshape(TMP_SPEED(a:b,:),[1 4*opts.window_size]))*opts.d_wheel/2*3.6 ...
                mean(abs(TMP_ACC_X(a:b))) mean(abs(TMP_ACC_Y(a:b)))];
            tmp_idx_speed =  double(enable(1) >= opts.min_speed);
            tmp_idx_acc_x =  double(enable(2) <= opts.max_acc_x);
            tmp_idx_acc_y =  double(enable(3) <= opts.max_acc_y);
            tmp_idx = floor((tmp_idx_speed+tmp_idx_acc_x+tmp_idx_acc_y)/3);
            idx(a:b,1) = tmp_idx;
            if tmp_idx == 0
                % observation doesn't comply with enable requirements
                % -> check observation starting from next time step
                a = a+1;
                continue
            end

            % observation complies with enable requirements
            % copy observation to data-struct
            for cntSens = 1 : length(fieldsInputData)
                if isempty(data.(fieldsInputData{cntSens}))
                    if ~isnan(inputData.(fieldsInputData{cntSens}))
                        data.(fieldsInputData{cntSens}) = inputData.(fieldsInputData{cntSens})(a:b,:)';
                    else
                        data.(fieldsInputData{cntSens}) = NaN(1,b-a+1);
                    end
                    continue
                end
                if ~isnan(inputData.(fieldsInputData{cntSens}))
                    data.(fieldsInputData{cntSens}) = [data.(fieldsInputData{cntSens});...
                        inputData.(fieldsInputData{cntSens})(a:b,:)'];
                else
                    data.(fieldsInputData{cntSens}) = [data.(fieldsInputData{cntSens}); NaN(1,b-a+1)];
                end     
            end

            % copy file properties to data-struct
            if istable(Label_Data)
                if isempty(data.Prop)
                    structFromLabelData = table2struct(Label_Data);

                    % Initialize Prop-Struct
                    f = [fieldnames(structFromLabelData)', fieldnames(dataProp)'];
                    f{2,1} = {[]};
                    data.Prop = struct(f{:});

                end

                % copy content of Label_Data to data.Prop-struct
                fieldnamesStructLabel = fieldnames(structFromLabelData)';
                for cntFieldnames = 1 : length(fieldnamesStructLabel)
                   data.Prop.(fieldnamesStructLabel{cntFieldnames}) = ...
                       [data.Prop.(fieldnamesStructLabel{cntFieldnames}); Label_Data.(fieldnamesStructLabel{cntFieldnames})];
                end
            end

            % copy content of dataProp to data.Prop-struct
            fieldnamesdataProp = fieldnames(dataProp)';
            
            % Initialize Property struct if it doesn't exist
            if ~isstruct(data.Prop)
                f = fieldnames(dataProp)';
                f{2,1} = {[]};
                data.Prop = struct(f{:});
            end
            
            for cntFieldnames = 1 : length(fieldnamesdataProp)
                if ~isempty(dataProp.(fieldnamesdataProp{cntFieldnames}))
                    if strcmp(fieldnamesdataProp{cntFieldnames}, 'timeStr')
                        idxStartProp = find(dataProp.timeNum(:,1) >= tDataInterp(a), 1, 'first');
                        idxStopProp = find(dataProp.timeNum(:,1) <= tDataInterp(b), 1, 'last');
                    else
                        idxStartProp = find(dataProp.(fieldnamesdataProp{cntFieldnames})(:,1) >= tDataInterp(a), 1, 'first');
                        idxStopProp = find(dataProp.(fieldnamesdataProp{cntFieldnames})(:,1) <= tDataInterp(b), 1, 'last');
                    end
                    % Prevent missing values due to unmatching time
                    % intervalls -> take last value of prop-variable
                    if isempty(idxStartProp)
                        idxStartProp = length(dataProp.(fieldnamesdataProp{cntFieldnames}));
                    end
                    data.Prop.(fieldnamesdataProp{cntFieldnames}) = ...
                       [data.Prop.(fieldnamesdataProp{cntFieldnames}); dataProp.(fieldnamesdataProp{cntFieldnames})(idxStartProp,2)];
                end
            end
            a = a+opts.window_size-opts.overlap;
        end
    end

    if isfield(data, 'SPEED_FL')
        data.Prop.meanWheelspeed = ...
            mean(data.SPEED_FL + data.SPEED_FR + data.SPEED_RL + data.SPEED_RR,2)/4;
    end
    if isfield(data, 'ACC_X')
        data.Prop.meanAx = mean(data.ACC_X,2);
    end
    if isfield(data, 'ACC_Y')
        data.Prop.meanAy = mean(data.ACC_Y,2);
    end
    if isfield(data, 'YAW_RATE')
        data.Prop.meanYawRate = mean(data.YAW_RATE,2);
    end
    
end

