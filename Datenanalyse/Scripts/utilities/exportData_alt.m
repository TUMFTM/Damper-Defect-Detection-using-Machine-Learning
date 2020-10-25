function [data_append] = exportData(data, opts, tf, ctrl)
% Script for Exporting the data as .csv to a selected folder, additionally
% writing information to Info.txt
% returns data struct, appended by Prop.labelInt

%% Create data structure for csv format (align all sensors column-wise)
    n_samples = size(data.(opts.fieldsData{1}),1); % number of samples (rows)
    export_data = zeros(n_samples,int32(opts.n_sensors*opts.t*opts.fs)); % preallocate memory
    for lng = 1:opts.n_sensors
        export_data(:,(1+(lng-1)*opts.t*opts.fs):(lng*opts.t*opts.fs)) = data.(opts.fieldsData{lng}); % concat data
    end

    %% mapping labels to integers
    data.Prop.labelInt = zeros(n_samples,1); % preallocate memory
    for ii=1:n_samples
       switch string(data.Prop.labelIsolation{ii})
           case 'good'
               data.Prop.labelInt(ii) = 0;
           case 'all'
               data.Prop.labelInt(ii) = 1;
           case 'FL'
               data.Prop.labelInt(ii) = 2;
           case 'RR'
               data.Prop.labelInt(ii) = 3; % RR instead of FR for consistency with DD dataset
           case 'FR'
               data.Prop.labelInt(ii) = 4;
           case 'RL'
               data.Prop.labelInt(ii) = 5;
           otherwise
               warning('Element %d yields incorrect label', ii)
               data.Prop.labelInt(ii) = 99;
       end
    end

    %% split data into training and test with random permutated indices 
    dataTraining = export_data(tf,:); %#ok<*NASGU>
    labelTraining = data.Prop.labelInt(tf,end); 
    dataTesting = export_data(~tf,:);
    labelTesting = data.Prop.labelInt(~tf,end);

    %% Write csv files
    disp('Select output folder for .csv data');
    folder_name = uigetdir();
    strings = {'dataTraining', 'dataTesting', 'labelTraining', 'labelTesting'};
    for ii = 1:length(strings)
        csvfilename = fullfile(folder_name, [strings{1,ii} '.csv']);
        csvwrite(csvfilename, eval(strings{1,ii}))
    end
    
    %% Write Info.txt
    fid = fopen([folder_name '/Info.txt'],'wt');
    fprintf(fid,'Total of %d sensor channels\n', opts.n_sensors);
    fprintf(fid,'Length per channel %f s (maps to %d datapoints)\n',opts.t,opts.t*opts.fs);
    fprintf(fid,'Alignment of sensor channels:\n');
    fprintf(fid,'%s\n', strjoin(opts.fieldsData)); 
    fprintf(fid,'Overlap between observations/sequences: %s datapoints\n', opts.overlap);
    fprintf(fid,'Label indices: 0: good, 1: all, 2: FL, 3:RR, 4:FR, 5:RL, 99: incorrect label (!)\n');
%     fprintf(fid,'Applied detrend? %s\n', ctrl.detrend_mode);
    fprintf(fid,'%d%% of dataset used as training, rest as test\n', opts.split*100);
    fclose(fid);
    
    %% return value
    data_append = data;

