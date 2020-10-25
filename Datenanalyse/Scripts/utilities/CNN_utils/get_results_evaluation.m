function [result_array, label, means, stds] = get_results_evaluation(folder_name)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
orig_folder = pwd;
cd(folder_name)
cv_logs = dir('**/CV_log.txt');
cd(orig_folder)
result_lines = cell(length(cv_logs),1);
result_strings = cell(length(cv_logs),1);
label = cell(length(cv_logs),1);
for ii=1:length(cv_logs)
    fid = fopen(fullfile(cv_logs(ii).folder, cv_logs(ii).name),'r');
    result_lines{ii} = fgetl(fid);
    fclose(fid);
    idxStart = strfind(result_lines{ii},'0.');
    result_lines{ii} = result_lines{ii}(idxStart(1):end-1);
    result_strings{ii} = regexp(result_lines{ii}, ', ', 'split');
    for jj=1:length(result_strings{ii})
        result_array(ii,jj) = str2double(result_strings{ii}{jj});
    end
    label{ii,1} = cv_logs(ii).folder(end-11:end-4);
end
means = mean(result_array, 2)*100;
stds = std(result_array,0,2)*100;
end

