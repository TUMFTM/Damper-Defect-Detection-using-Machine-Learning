%% Save as PDF
function [] = SaveAsPDFs()
    % Current figure
    if isempty(findobj('Type','Figure')), return, end
    hFig = gcf;
    % Get path to save
    [filename, pathname, filterindex] = uiputfile({'*.jpg;*.png;*.eps;*.emf;*.pdf', 'Image (*.jpg,*.png,*.eps,*.emf,*.pdf)'},...
                                                    'Aktuelle ansicht speichern unter...');  
    % Hide all objects, that are no axes
    objs = findobj(gcf,'-not','Type','axes','-and','-not','Type','legend','-depth',1);
    if filterindex ~=1
        set(objs,'Visible','off')  
    end         
    % Save
    if filename ~= 0
        if strfind(filename,'pdf')
            if ~isempty(hFig), f = hFig; else, f = gcf; end
            child = allchild(f);
            h = figure('OuterPosition',get(0, 'Screensize'));
            warning('off','MATLAB:copyobj:ObjectNotCopied')
            copyobj(child,h);
            warning('on','MATLAB:copyobj:ObjectNotCopied')
            set(h,'Colormap',get(f,'Colormap'));
            set(h, 'PaperUnits','centimeters');
            set(h, 'Units','centimeters');
            pos=get(h,'Position');
            set(h, 'PaperSize', [pos(3) pos(4)]);
            set(h, 'PaperPositionMode', 'manual');
            set(h, 'PaperPosition',[0 0 pos(3) pos(4)]);
            set(h, 'InvertHardCopy','on');
            saveas(h,[pathname filesep filename]);
            close(h)
        else
            saveas(hFig,[pathname filesep filename]);
        end
    end
    % Show again all objects, that are no axes
    set(objs,'Visible','on') 
end

