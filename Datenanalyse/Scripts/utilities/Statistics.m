%% Get data
min_score = 0.1;

if exist('score') ==1 && isfield(observation_properties,'score') == 0 
    observation_properties = [observation_properties table(score)];
end

if exist('validation') ==1 && isfield(observation_properties,'validation') == 0 
    observation_properties = [observation_properties table(validation)];
end

%% General dataset plots
set(figure,'Name','General dataset and track information','NumberTitle','off');
subplot(2,2,1);
cat_detection = categorical(observation_properties.Label_DETECTION,{'fault' 'good'},'Ordinal',true);
histogram(cat_detection);
title('Detection classes')
ylabel('number of observations')

subplot(2,2,2);
cat_isolation = categorical(observation_properties.Label_ISOLATION,{'good' 'all' 'FL' 'FR' 'RL' ...
    'RR'},'Ordinal',true);
histogram(cat_isolation);
title('Isolation & detection classes')
ylabel('number of observations')

subplot(2,2,3);
cat_track = categorical(observation_properties.track,{'A9/A92' 'A9' 'A92' 'Landstraﬂe' ...
    'Landstraﬂe Freising' 'Freising Kopfsteinpflaster' 'Schlechtwegstrecke 1' ...
    'Schlechtwegstrecke 2' 'Other'},'Ordinal',true);
histogram(cat_track);
title('Tracks')
ylabel('number of observations')

subplot(2,2,4);
cat_track_condition = categorical(observation_properties.track_condition,{'glatt' 'rau' 'glatt/rau'},...
    'Ordinal',true);
histogram(cat_track_condition);
title('Track conditions');
ylabel('number of observations')

set(figure,'Name','Mean speed and acceleration of observations','NumberTitle','off');
subplot(2,2,1);
h = histogram(observation_properties.mean_speed);
h.BinWidth = 1;
title('Mean speed');
xlabel('km/h')
ylabel('number of observations')

subplot(2,2,2);
h = histogram(observation_properties.mean_acc_x);
h.BinWidth = 0.025;
title('Mean acceleration_x');
xlabel('m/s^2')
ylabel('number of observations')

subplot(2,2,3);
h = histogram(observation_properties.mean_acc_y);
h.BinWidth = 0.025;
title('Mean acceleration_y');
xlabel('m/s^2')
ylabel('number of observations')

%% Isolated dataset information | high vs. low confidence level
rows = observation_properties.score < min_score;

set(figure,'Name','Comparison of high and low confidence level data','NumberTitle','off');
subplot(1,2,1);
cat_track_iso = categorical(observation_properties.track,{'A9/A92' 'A9' 'A92' 'Landstraﬂe' ...
    'Landstraﬂe Freising' 'Freising Kopfsteinpflaster' 'Schlechtwegstrecke 1' ...
    'Schlechtwegstrecke 2' 'Other'},'Ordinal',false);
h1 = histogram(cat_track_iso(~rows));
h1.FaceColor = 'gr';
hold on
h2 = histogram(cat_track_iso(rows));
h2.FaceColor = 'r';
title('Tracks')
ylabel('number of observations')
legend('high confidence level','low confidence level')

subplot(1,2,2);
cat_track_condition_iso = categorical(observation_properties.track_condition,{'glatt' 'rau' 'glatt/rau'},...
    'Ordinal',false);
h1 = histogram(cat_track_condition_iso(~rows));
h1.FaceColor = 'gr';
hold on
h2 = histogram(cat_track_condition_iso(rows));
h2.FaceColor = 'r';
title('Track conditions')
ylabel('number of observations')
legend('high confidence level','low confidence level')

set(figure,'Name','Comparison of high and low confidence level data','NumberTitle','off');
subplot(2,2,1);
h1 = histogram(observation_properties.mean_speed(~rows));
h1.BinWidth = 1;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_speed(rows));
h2.BinWidth = 1;
h2.FaceColor = 'r';
title('Mean speed');
xlabel('km/h')
ylabel('number of observations')
legend('high confidence level','low confidence level')

subplot(2,2,2);
h1 = histogram(observation_properties.mean_acc_x(~rows));
h1.BinWidth = 0.025;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_acc_x(rows));
h2.BinWidth = 0.025;
h2.FaceColor = 'r';
title('Mean acceleration_x');
xlabel('m/s^2')
ylabel('number of observations')
legend('high confidence level','low confidence level')

subplot(2,2,3);
h1 = histogram(observation_properties.mean_acc_y(~rows));
h1.BinWidth = 0.025;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_acc_y(rows));
h2.BinWidth = 0.025;
h2.FaceColor = 'r';
title('Mean acceleration_y');
xlabel('m/s^2')
ylabel('number of observations')
legend('high confidence level','low confidence level')

%% Isolated dataset information | right vs. wrong classification
rows = observation_properties.validation < 1;

set(figure,'Name','Comparison of right and wrong classification data','NumberTitle','off');
subplot(1,2,1);
cat_track_iso = categorical(observation_properties.track,{'A9/A92' 'A9' 'A92' 'Landstraﬂe' ...
    'Landstraﬂe Freising' 'Freising Kopfsteinpflaster' 'Schlechtwegstrecke 1' ...
    'Schlechtwegstrecke 2' 'Other'},'Ordinal',false);
h1 = histogram(cat_track_iso(rows));
h1.FaceColor = 'gr';
hold on
h2 = histogram(cat_track_iso(~rows));
h2.FaceColor = 'r';
title('Tracks')
ylabel('number of observations')
legend('right classification','wrong classification')

subplot(1,2,2);
cat_track_condition_iso = categorical(observation_properties.track_condition,{'glatt' 'rau' 'glatt/rau'},...
    'Ordinal',false);
h1 = histogram(cat_track_condition_iso(rows));
h1.FaceColor = 'gr';
hold on
h2 = histogram(cat_track_condition_iso(~rows));
h2.FaceColor = 'r';
title('Track conditions')
ylabel('number of observations')
legend('right classification','wrong classification')

set(figure,'Name','Comparison of right and wrong classification data','NumberTitle','off');
subplot(2,2,1);
h1 = histogram(observation_properties.mean_speed(rows));
h1.BinWidth = 1;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_speed(~rows));
h2.BinWidth = 1;
h2.FaceColor = 'r';
title('Mean speed');
xlabel('km/h')
ylabel('number of observations')
legend('right classification','wrong classification')

subplot(2,2,2);
h1 = histogram(observation_properties.mean_acc_x(rows));
h1.BinWidth = 0.025;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_acc_x(~rows));
h2.BinWidth = 0.025;
h2.FaceColor = 'r';
title('Mean acceleration_x');
xlabel('m/s^2')
ylabel('number of observations')
legend('right classification','wrong classification')

subplot(2,2,3);
h1 = histogram(observation_properties.mean_acc_y(rows));
h1.BinWidth = 0.025;
h1.FaceColor = 'gr';
hold on
h2 = histogram(observation_properties.mean_acc_y(~rows));
h2.BinWidth = 0.025;
h2.FaceColor = 'r';
title('Mean acceleration_y');
xlabel('m/s^2')
ylabel('number of observations')
legend('right classification','wrong classification')

