function [data_append] = exportDataToCsv(data, opts)
% Script for Exporting the data as .csv to a selected folder, additionally
% writing information to Info.txt
% returns data struct, appended by Prop.labelInt

%% Create data structure for csv format (align all sensors column-wise)
    n_samples = size(data.(opts.fieldsData{1}),1); % number of samples (rows)
    export_data = zeros(n_samples,int32(opts.n_sensors*opts.t*opts.fs)); % preallocate memory
    for lng = 1:opts.n_sensors
        export_data(:,(1+(lng-1)*opts.t*opts.fs):(lng*opts.t*opts.fs)) = data.(opts.fieldsData{lng}); % concat data
    end

    %% split data into training and test with random permutated indices 
    dataset = export_data; 
    labels = data.Prop.LabelID; 
    obsID = data.Prop.observationID;

    %% Write csv files   
    disp('Select output folder for .csv files');
    folder_name = uigetdir();
    strings = {'dataset', 'labels', 'obsID'};
    for ii = 1:length(strings)
        csvfilename = fullfile(folder_name, [strings{1,ii} '.csv']);
        csvwrite(csvfilename, eval(strings{1,ii}))
    end
    %% Write Info.txt
    fid = fopen([folder_name '/Info.txt'],'wt');
    fprintf(fid,'Total of %d sensor channels\n', opts.n_sensors);
    fprintf(fid,'Length per channel %f s\n',opts.t);
    fprintf(fid,'Number of datapoints: %d\n', opts.t*opts.fs);
    fprintf(fid,'Alignment of sensor channels:\n');
    fprintf(fid,'%s\n', strjoin(opts.fieldsData));
    fprintf(fid,'Overlap between observations/sequences: %d datapoints\n', opts.overlap);
    fprintf(fid,'Label indices: See global label mapping, LabelID.xlsx\n');
    fprintf(fid,'Applied detrend? %d\n', opts.detrend);
    if opts.detrend
        fprintf(fid,'Detrend applied: %s\n', opts.detrend_mode);
    end
    fprintf(fid,'Sensor channel names:\n');
    for cntRows = 1 : size(opts.sensor_names,1)
        fprintf(fid,'%s\n', strjoin(opts.sensor_names(cntRows,:)));
    end
    
    % Print additional information
    fprintf(fid, '\nAdditional information\n');
    fieldsOpts = fields(opts);
    for cntFields = 1 : length(fieldsOpts)
        if isnumeric(opts.(fieldsOpts{cntFields}))
            fprintf(fid, '%s: %d\n', fieldsOpts{cntFields}, opts.(fieldsOpts{cntFields}));
        elseif ischar(opts.(fieldsOpts{cntFields}))
            fprintf(fid, '%s: %s\n', fieldsOpts{cntFields}, opts.(fieldsOpts{cntFields}));
        end
    end
    
    fclose(fid);
    
    %% return value
    data_append = data;

