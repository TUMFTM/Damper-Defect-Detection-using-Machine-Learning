function [table_out] = rearrangeFeatureTable(table_in)
%REARRANGEFEATURETABLE 
%   Rearrange the feature table from feature blocks within the table to
%   single features within the table

    fprintf('\nRearranging Feature Table...');

    table_out = table();
    
    addLabel = 0;
    % Extract label as it should keep its name
    if max(strcmp(fields(table_in),'Label'))
        Label = table_in.Label;
        table_in.Label = [];
        addLabel = 1;
    elseif max(strcmp(fields(table_in),'Label_ISOLATION'))
        Label = table_in.Label_ISOLATION;
        table_in.Label_ISOLATION = [];
        addLabel = 1;
    end

    for cnt_FeatBlock = 1 : size(table_in,2)
        
        tmpDATA_classification = table(table2array(table_in(:,cnt_FeatBlock)));
        tmpFeatBlockDATA_classification = table2array(tmpDATA_classification);
        
        for cnt_subFeat = 1 : size(tmpFeatBlockDATA_classification,2)
            
            tmpSubFeat = table(tmpFeatBlockDATA_classification(:,cnt_subFeat));
            tmpSubFeat.Properties.VariableNames = ...
                {[table_in.Properties.VariableNames{cnt_FeatBlock},'_',num2str(cnt_subFeat)]};
            table_out = [table_out tmpSubFeat];
            clear tmpSubFeat;
        end
        
    end

    if addLabel == 1
        table_out.Label = Label;
    end
    
    fprintf('finished\n');
    
end

