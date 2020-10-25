function [result_array, label, means, stds] = get_results_plainRun(folder_name)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    logs = dir([folder_name, '/**/log.txt']);

    result_array = zeros(length(logs),1);
    label = cell(length(logs),1);
    for ii=1:length(logs)
        
        % Read file and close it again
        fid = fopen(fullfile(logs(ii).folder, logs(ii).name),'r');
        txtData = textscan(fid,'%[^\n\r]');
        fclose(fid);
        
        % Extract Accuracy
        txtData = txtData{:};
        idxLine = strfind(txtData,'Accuracy:	0.');
        idxLine = ~cellfun(@isempty,idxLine);
        if sum(idxLine)==0
            fprintf('hier ist was schief gelaufen');
        end
        tmp_result = textscan(txtData{idxLine},'%s%f');
        result_array(ii,1) = tmp_result{1,2};
        
        % Save label 'e00x_k00x'
        tmp_label = split(logs(ii).folder, '\');
        label{ii,1} = tmp_label{end};

    end
    means = mean(result_array, 2)*100;
    stds = std(result_array,0,2)*100;
end

