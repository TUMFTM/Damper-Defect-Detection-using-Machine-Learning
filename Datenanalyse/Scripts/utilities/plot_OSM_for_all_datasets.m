% Draw Map for full dataset
x = [11.1 11.1 12.1 12.1];
y = [48.2 48.8 48.2 48.8];
hFigure = figure('Units', 'centimeters', 'Position', [9 2 24 24]); 
hLine = plot(x, y, 'o-', 'LineWidth', 2);
hBase = plot_openstreetmap('Scale', 2);
predFigure = plotPredictionAnalysisMap(data, 'figure', hFigure);
delete(hLine)
xlabel('Longitudinal GPS coordinate in deg');ylabel('Latitudinal GPS coordinate in deg');
print('OSM_FullDataset','-dpng','-r300')

% Draw map for mass dataset
x = [11.1 11.1 12.1 12.1];
y = [48 48.8 48 48.8];
hFigure = figure('Units', 'centimeters', 'Position', [9 2 24 24]); 
hLine = plot(x, y, 'o-', 'LineWidth', 2);
hBase = plot_openstreetmap('Scale', 2);
predFigure = plotPredictionAnalysisMap(testDD2Mass.data, 'figure', hFigure);
delete(hLine)
xlabel('Longitudinal GPS coordinate in deg');ylabel('Latitudinal GPS coordinate in deg');
print('OSM_MassDataset','-dpng','-r300')

x = [11.5 11.5 12.9 12.9];
y = [47.5 48.35 47.5 48.35];
hFigure = figure('Units', 'centimeters', 'Position', [9 2 24 24]); 
hLine = plot(x, y, 'o-', 'LineWidth', 2);
hBase = plot_openstreetmap('Scale', 2);
predFigure = plotPredictionAnalysisMap(testDD2Tire.data, 'figure', hFigure);
delete(hLine)
ylim([47.4 48.45])
xlabel('Longitudinal GPS coordinate in deg');ylabel('Latitudinal GPS coordinate in deg');
print('OSM_TireDataset','-dpng','-r300')