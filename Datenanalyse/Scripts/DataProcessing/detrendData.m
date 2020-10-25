function data_out = detrendData(data_in, opts, varargin)
% Detrending data

    if find(strcmp(varargin,'displayText'))
        ctrl.displayText = varargin{find(strcmp(varargin,'displayText'))+1};
    else
        ctrl.displayText = 1;
    end

    props = data_in.Prop; % backup property struct
    if opts.detrend
        switch opts.detrend_mode
            case 'linear'
                txt = ('detrend_mode set to linear, Applying DETREND algorithm to data');
                for cnt_fields = 1 : length(opts.fieldsData)
                    data_out.(opts.fieldsData{cnt_fields}) = detrend(data_in.(opts.fieldsData{cnt_fields})')';
                end
            case 'const'
                txt = ('detrend_mode set to const, removing constant mean from data');
                for cnt_fields = 1 : length(opts.fieldsData)
                    data_out.(opts.fieldsData{cnt_fields}) = detrend(data_in.(opts.fieldsData{cnt_fields})','constant')';
                end    
            case 'standardize'
                txt = ('detrend_mode set to standardize, Applying mean 0 and std_dev 1 to data');
                for cnt_fields = 1 : length(opts.fieldsData)
                    data_out.(opts.fieldsData{cnt_fields}) = zscore(data_in.(opts.fieldsData{cnt_fields})')';
                end
            case 'movmean'
                txt = ('detrend_mode set to moving average, removing moving average of 21 (10 before + 1 actual + 10 after) samples from data');
                for cnt_fields = 1 : length(opts.fieldsData)
                    data_out.(opts.fieldsData{cnt_fields}) = data_in.(opts.fieldsData{cnt_fields}) - movmean(data_in.(opts.fieldsData{cnt_fields}), [10 10], 2);
                end
            otherwise
                txt = ('detrend_mode not set properly, skipping detrend of data');
        end
    else
        txt = ('no detrend applied');
        data_out = data_in;
    end
    data_out.Prop = props; % restore property struct
    
    if ctrl.displayText
        fprintf(txt);
    end
    
end