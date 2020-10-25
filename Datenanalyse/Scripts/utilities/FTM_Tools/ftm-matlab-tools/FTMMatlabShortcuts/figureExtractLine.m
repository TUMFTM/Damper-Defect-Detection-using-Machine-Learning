function [] = ExtractLine()
    % Find Lines 
    if isempty(findobj('Type','Figure')), return, end
    axis  = findobj(gcf,'Type','Axes','Selected', 'on');
    if isempty(axis), axis = gca; end
    lines = findobj(gcf,'Type','Line','Selected', 'on');
    for i=1:length(axis)
        linesAx = findobj(axis(i), 'Type', 'Line');
        lines = [lines; linesAx];
    end   
    if isempty(lines) && isempty(axis)
        waitfor(warndlg('No line selected or found in axis.', 'Warning...'))
        return
    end
    lines=unique(lines);
    for i=1:length(lines)
        handles{i,1} = lines(i);
        handles{i,2} = get(get(get(lines(i),'Parent'),'Title'),'String');
        if isempty(handles{i,2}) || ~isvarname(handles{i,2})
            if ~exist('allAxis'), allAxis = findobj(gcf, 'Type', 'Axes'); end
            for j=1:length(allAxis)
                if all(get(allAxis(j),'Position') == get(get(lines(i),'Parent'),'Position'))
                    handles{i,2} = ['axis', num2str(length(allAxis)-j+1)];
                    continue
                end
            end
        end
    end
    % Name of Structure for saving
    nameFig = get(gcf, 'Name');
    numFig  = get(gcf, 'Number');
    if ~isvarname(nameFig) || ismember(nameFig,evalin('base','who'))
        nameFig = ['figure',num2str(numFig)]; 
        indexRun = 1;
        while ismember(nameFig,evalin('base','who'))
            nameFig = ['figure',num2str(numFig),num2str(indexRun)]; 
            indexRun = indexRun+1;
        end
    end
        
    % Write Data
    for j=1:size(handles,1)
        if isvarname(get( handles{j,1}, 'DisplayName'))
            lineName = get( handles{j,1}, 'DisplayName');
        else
            lineName = 'line1';
            if j > 1 && isfield(eval(nameFig), handles{j,2}) 
                index = 1;
                lineNameOrig = lineName(1:end-1);
                while ismember(lineName,fieldnames(eval([nameFig,'.',handles{j,2}])))
                    lineName = [lineNameOrig, num2str(index)];
                    index = index+1;
                end 
            end
        end
        eval([nameFig,'.',handles{j,2},'.',lineName,'.XData = lines(j).XData;'])
        eval([nameFig,'.',handles{j,2},'.',lineName,'.YData = lines(j).YData;'])
    end
    % Save
    nameFields = fieldnames(eval(nameFig));
    if length(nameFields)==1 && ~ismember(nameFields{1},evalin('base','who'))
        nameSave = fieldnames(eval(nameFig));
        nameSave = nameSave{1};
        assignin('base',nameSave, eval([nameFig,'.',nameSave]))     
    else
        assignin('base',nameFig, eval(nameFig))
        nameSave = nameFig;
    end
    
    waitfor(msgbox(['Gespeichert als: ',nameSave],'Info...','help'))
end

