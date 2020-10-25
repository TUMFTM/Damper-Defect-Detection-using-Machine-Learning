function [here] = addHereDataToProp(dataProp,varargin)
%ADDHEREDATATOPROP Adds data from here to the Prop-Field of the data-struct
% Example data.Prop.here = addHereDataToProp(data.Prop)
    
    
    addpath(genpath('C:\Users\tzehe\Documents\Fahrwerkdiagnose\here'));

    if find(strcmp(varargin,'figureName'))
        idxVararginFigureName = find(strcmp(varargin,'figureName'));
        ctrl.figureName = varargin{idxVararginFigureName+1};
        varargin{idxVararginFigureName:idxVararginFigureName+1} = [];
    else
        ctrl.figureName = '';
    end
    
    %% Prepare Request
    % Here credentials
    [appID,appCode] = getHereCredentials();
    hereAuthentication = ['?app_id=' appID '&app_code=' appCode];
    % Here API    
    hereAPI = 'https://rme.api.here.com/2/matchroute.json';
    % Routing mode
    routingMode = ['&routemode=car'];
    
%     [linkIDFilename, linkIDPath] = uigetfile('*.mat','Please select LinkID-file');
    linkIDFilename = 'DD2_DD2Mass_DD2Tire.mat';
    linkIDPath = 'C:\Users\tzehe\Documents\Fahrwerkdiagnose\here\hereData';
    load(fullfile(linkIDPath, linkIDFilename));
    
    %% Build trace file and Request
    for iElement=1:size(dataProp.GPS_LONG,1)
        if ~isnan(dataProp.GPS_LONG(iElement))
            txt = sprintf('LONGITUDE,LATITUDE,HEADING\n');
            for iPoint = 1:size(dataProp.GPS_LAT,2)
                txt = [txt,sprintf('\n%f,%f,%f',dataProp.GPS_LONG(iElement,iPoint),dataProp.GPS_LAT(iElement,iPoint),dataProp.GPS_HEADING(iElement,iPoint))];
            end
            txt = uint8(txt);
            % Encode base64
            txt = matlab.net.base64encode(txt);
            traceFile = ['&file=',txt];
            % Construct Url
            urlMapMatching = [hereAPI hereAuthentication routingMode traceFile];

            % Request and decode
            [hereResponseAPI, requestStatus] = urlread(urlMapMatching);
%             if requestStatus==0, dataOutput=[]; return, end
            if requestStatus==0
                dataOutput{iElement} = NaN;
                continue;
            end
            dataOutput{iElement} = rebuildData(jsondecode(hereResponseAPI));
        else
            dataOutput{iElement} = NaN;
        end
    end
    
    
    fig = figure('Name',ctrl.figureName); hold on, ax1 = fig.CurrentAxes; c1 = colorbar(); cmap = colormap('jet');
    fig = figure('Name',ctrl.figureName); hold on, ax2 = fig.CurrentAxes; c2 = colorbar(); cmap = colormap('jet'); 

    cminmax1 = [0, max([linkData.fromAvgIRI;linkData.toAvgIRI])]; caxis(ax1, cminmax1);
    cminmax2 = [1, 3]; caxis(ax2, cminmax2); c2.TicksMode = 'manual'; c2.TickLabelsMode = 'manual'; c2.Ticks = [1 2 3]; c2.TickLabels = {'Good','Fair','Poor'}; caxis(ax2, cminmax2);

    
    here = struct();
    
    for iObservation=1:length(dataOutput)
        try
            linkIDs = unique(dataOutput{iObservation}.linkId);    
        %     plot(ax, mean(dataOutput{iObservation}.lonMatched), mean(dataOutput{iObservation}.latMatched), 'LineStyle','none', 'Color','k','Marker', 'o', 'MarkerSize', 10)   

            for iLink=1:length(linkIDs)
                indexID = find(linkData.id == abs(linkIDs(iLink)));
                isInvalid = false;
                % Check if IRI is valid
                if strcmp(linkData.travelDirection{indexID},'F') && linkData.fromAvgIRI(indexID)==-1
                    isInvalid = true;
                elseif strcmp(linkData.travelDirection{indexID},'T') && linkData.toAvgIRI(indexID)==-1
                    isInvalid = true;
                elseif strcmp(linkData.travelDirection{indexID},'B') && (linkData.fromAvgIRI(indexID)==-1 || linkData.toAvgIRI(indexID)==-1 )
                    if linkData.fromAvgIRI(indexID)==-1
                        isInvalid = true;
                    elseif linkData.toAvgIRI(indexID)==-1
                        isInvalid = true;
                    end
                end
                if isInvalid
                    addLink2Plot(ax1, linkData.shape{indexID},[1 0 1],'-',2);
                else
                    if linkIDs(iLink) > 0%strcmp(linkData.travelDirection{indexID}, 'F')
                        iri = linkData.fromAvgIRI(indexID);
                    elseif linkIDs(iLink) < 0%strcmp(linkData.travelDirection{indexID}, 'T')
                        iri = linkData.toAvgIRI(indexID);
        %             elseif strcmp(linkData.travelDirection{indexID}, 'B')
        %                 if linkData.fromAvgIRI(indexID)==-1 
        %                     iri = linkData.toAvgIRI(indexID);
        %                 elseif linkData.toAvgIRI(indexID)==-1
        %                     iri = linkData.fromAvgIRI(indexID);
        %                 else
        %                     iri = mean([linkData.fromAvgIRI(indexID),linkData.toAvgIRI(indexID)]);
        %                 end
                    end
                    color = [interp1(linspace(max(cminmax1(1),0),max(cminmax1(end),1),size(cmap,1)),cmap(:,1),iri,'linear'),... 
                             interp1(linspace(max(cminmax1(1),0),max(cminmax1(end),1),size(cmap,1)),cmap(:,2),iri,'linear'),...
                             interp1(linspace(max(cminmax1(1),0),max(cminmax1(end),1),size(cmap,1)),cmap(:,3),iri,'linear')];
                    if ~any(isnan(color))
                        addLink2Plot(ax1, linkData.shape{indexID},color,'-',2,sprintf('disp(''%s'')',dataProp.labelIsolation{iObservation}));
                    end
                    % Category
                    if linkIDs(iLink)>0%strcmp(linkData.travelDirection{iLink}, 'F')
                        cat = linkData.fromAvgRoughnessCat(indexID);
                    elseif linkIDs(iLink)<0%strcmp(linkData.travelDirection{iLink}, 'T')
                        cat = linkData.toAvgRoughnessCat(indexID);
                    elseif strcmp(linkData.travelDirection{iLink}, 'B')
        %                 if linkData.fromAvgRoughnessCat(iLink)==-1 
        %                     cat = linkData.toAvgRoughnessCat(iLink);
        %                 elseif linkData.toAvgRoughnessCat(iLink)==-1
        %                     cat = linkData.fromAvgRoughnessCat(iLink);
        %                 else
        %                     cat = mean([linkData.fromAvgRoughnessCat(iLink),linkData.toAvgRoughnessCat(iLink)]);
        %                 end
                    end
                    color = [interp1(linspace(max(cminmax2(1),1),max(cminmax2(end),3),size(cmap,1)),cmap(:,1),cat,'linear'),... 
                             interp1(linspace(max(cminmax2(1),1),max(cminmax2(end),3),size(cmap,1)),cmap(:,2),cat,'linear'),...
                             interp1(linspace(max(cminmax2(1),1),max(cminmax2(end),3),size(cmap,1)),cmap(:,3),cat,'linear')];
                    if ~any(isnan(color))
                        addLink2Plot(ax2, linkData.shape{indexID},color,'-',2);
                    end

                end

                if iObservation == 1
                    here.AvgIRI = iri;
                    here.RoughnessCat = cat;
                else
                    here.AvgIRI = [here.AvgIRI; iri];
                    here.RoughnessCat = [here.RoughnessCat; cat];
                end

            end
        catch
            if iObservation == 1
                here.AvgIRI = NaN;
                here.RoughnessCat = NaN;
            else
                here.AvgIRI = [here.AvgIRI; NaN];
                here.RoughnessCat = [here.RoughnessCat; NaN];
            end
        end
    end
    
    if ~isfield(here, 'AvgIRI')
        here.AvgIRI = NaN .* ones(size(dataProp.GPS_LONG,1),1);
        here.RoughnessCat = NaN .* ones(size(dataProp.GPS_LONG,1),1);
    end
    
end


function [data] = rebuildData(dataIn)
    % Check if RouteLinks is cell
    if ~iscell(dataIn.RouteLinks), help = dataIn.RouteLinks; dataIn.RouteLinks = []; dataIn.RouteLinks{1} = help; end
    if length(dataIn.RouteLinks{1})>1
        for iStruct=length(dataIn.RouteLinks{1}):-1:1
            dataIn.RouteLinks{iStruct} = dataIn.RouteLinks{1}(iStruct);
        end
    end
    % Go through data
    for iTracePoint = 1:length(dataIn.TracePoints)
        data.confidenceValue(iTracePoint)=dataIn.TracePoints(iTracePoint).confidenceValue;
        data.elevation(iTracePoint)=dataIn.TracePoints(iTracePoint).elevation;
        data.headingDegreeNorthClockwise(iTracePoint)=dataIn.TracePoints(iTracePoint).headingDegreeNorthClockwise;
        data.headingMatched(iTracePoint)=dataIn.TracePoints(iTracePoint).headingMatched;
        data.lat(iTracePoint)=dataIn.TracePoints(iTracePoint).lat;
        data.latMatched(iTracePoint)=dataIn.TracePoints(iTracePoint).latMatched;
        data.linkIdMatched(iTracePoint)=dataIn.TracePoints(iTracePoint).linkIdMatched;
        data.lon(iTracePoint)=dataIn.TracePoints(iTracePoint).lon;
        data.lonMatched(iTracePoint)=dataIn.TracePoints(iTracePoint).lonMatched;
        data.matchDistance(iTracePoint)=dataIn.TracePoints(iTracePoint).matchDistance;
        data.matchOffsetOnLink(iTracePoint)=dataIn.TracePoints(iTracePoint).matchOffsetOnLink;
        data.minError(iTracePoint)=dataIn.TracePoints(iTracePoint).minError;
        data.routeLinkSeqNrMatched(iTracePoint)=dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched;
        data.speedMps(iTracePoint)=dataIn.TracePoints(iTracePoint).speedMps;
        data.timestamp(iTracePoint)=dataIn.TracePoints(iTracePoint).timestamp;
        data.linkId(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.linkId;
        data.functionalClass(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.functionalClass;
        data.confidence(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.confidence;
        data.linkLength(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.linkLength;
        data.mSecToReachLinkFromStart(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.mSecToReachLinkFromStart;
        data.shape{iTracePoint}=str2num(dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.shape);
        if isfield('offset',dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1})
            data.offset(iTracePoint)=dataIn.RouteLinks{dataIn.TracePoints(iTracePoint).routeLinkSeqNrMatched+1}.offset;
        else
            data.offset(iTracePoint) = 0;
        end
    end
end