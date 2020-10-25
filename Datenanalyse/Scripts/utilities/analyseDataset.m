function analyseDataset(data, opts)
    %ANALYSEDATASET Summary of this function goes here
    %   Script for visualisation of dataset characteristics

    %% Analyse classes
    classes = categorical(data.Prop.labelIsolation);
    
    figure;
    h = histogram(classes,'BarWidth',0.7,'DisplayOrder','descend');
    grid on
    xlabel('Class')
    ylabel('Number of Observations');
    
    % Add number of elements to figure
    barLength = h.Values;
    verticalOffset = 0.04*max(barLength);
    for cntBars = 1:length(barLength)
        text(cntBars,barLength(cntBars)+verticalOffset,num2str(barLength(cntBars)),'HorizontalAlignment','center')
    end

    
    %% Analyse tracks
    if isfield(data.Prop, 'track')
        tracks = categorical(data.Prop.track);

        figure;
        h = histogram(tracks,'BarWidth',0.7,'DisplayOrder','descend');
        grid on
        xlabel('Track')
        ylabel('Number of Observations');

        % Add number of elements to figure
        barLength = h.Values;
        verticalOffset = 0.04*max(barLength);
        for cntBars = 1:length(barLength)
            text(cntBars,barLength(cntBars)+verticalOffset,num2str(barLength(cntBars)),'HorizontalAlignment','center')
        end
    end

    
    %% Analyse track conditions
    if isfield(data.Prop, 'track')
        trackCondition = categorical(data.Prop.track);

        figure;
        h = histogram(trackCondition,'BarWidth',0.7,'DisplayOrder','descend');
        grid on
        xlabel('Track Condition')
        ylabel('Number of Observations');

        % Add number of elements to figure
        barLength = h.Values;
        verticalOffset = 0.04*max(barLength);
        for cntBars = 1:length(barLength)
            text(cntBars,barLength(cntBars)+verticalOffset,num2str(barLength(cntBars)),'HorizontalAlignment','center')
        end
    end
    
    %% Calculate time and distance
    travelledTime_sec = size(data.(opts.fieldsData{1}),2)*size(data.(opts.fieldsData{1}),1)/opts.fs;
    
	if isfield(data.Prop, 'meanWheelspeed')
        mean_Wheelspeed = data.Prop.meanWheelspeed;
        travelledDistance_km = sum(mean_Wheelspeed/3.6*size(data.(opts.fieldsData{1}),2)/opts.fs/1000);
    else
        travelledDistance_km = sum(mean(data.SPEED_FL,2).*opts.d_wheel/2*size(data.(opts.fieldsData{1}),2)/opts.fs)/1000;
    end
    
    fprintf('Travelled time in min: %.2f\n', travelledTime_sec/60);
    fprintf('Travelled distance in km: %.2f\n', travelledDistance_km);

end

