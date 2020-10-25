function [result_array, label, means, stds, time] = get_results(folder_name)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% orig_folder = pwd;
% cd(folder_name)
cv_logs = dir([folder_name, '/**/CV_log.txt']);
% cd(orig_folder)
result_lines = cell(length(cv_logs),1);
result_strings = cell(length(cv_logs),1);
label = cell(length(cv_logs),1);
time = zeros(length(cv_logs),1);

for ii=1:length(cv_logs)
    fid = fopen(fullfile(cv_logs(ii).folder, cv_logs(ii).name),'r');
    result_lines{ii} = fgetl(fid);
    tmp_time = fgetl(fid); % train accuracies verwerfen
    tmp_time = fgetl(fid); % last train accuracies verwerfen
    tmp_time = fgetl(fid); % time verwenden
    if strfind(tmp_time,'average training time: ')
        time(ii) = str2double(tmp_time(strfind(tmp_time,'average training time: ')+23:end));
    end
    fclose(fid);
    idxStart = strfind(result_lines{ii},'0.');
    result_lines{ii} = result_lines{ii}(idxStart(1):end-1);
    result_strings{ii} = regexp(result_lines{ii}, ', ', 'split');
    for jj=1:length(result_strings{ii})
        result_array(ii,jj) = str2double(result_strings{ii}{jj});
    end
%     tmp1 = cv_logs(ii).folder(end-11:end-4);
    tmp = strsplit(cv_logs(ii).folder, filesep);
    label{ii,1} = tmp{end};
end

means = mean(result_array, 2)*100;
stds = std(result_array,0,2)*100;
end

