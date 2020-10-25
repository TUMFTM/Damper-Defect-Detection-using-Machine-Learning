function [auc, fpr, tpr] = noveltyAUC(noveltyScore, trueLabels, intactClass)
%NOVELTYAUC Summary of this function goes here
%   Detailed explanation goes here

    all = table;
    all.noveltyScore = noveltyScore;
    all.trueLabels = trueLabels;

    all_sorted = sortrows(all,'noveltyScore');

    thresholds = flipud(unique(all_sorted.noveltyScore, 'sorted'));

    fpr = zeros(size(thresholds));
    tpr = zeros(size(thresholds));

    sum_true_positives = sum(~strcmp(trueLabels,intactClass));
    sum_true_negatives = sum(strcmp(trueLabels,intactClass));

    for cntidxT = 1 : length(thresholds)
        curr_thres = thresholds(cntidxT);
        idxPositive = all_sorted.noveltyScore > curr_thres;
        tpr(cntidxT) = sum(~strcmp(all_sorted.trueLabels(idxPositive),intactClass))/sum_true_positives;
        fpr(cntidxT) = sum(strcmp(all_sorted.trueLabels(idxPositive),intactClass))/sum_true_negatives;
    end

    auc = trapz(fpr,tpr);

    % figure;
    % plot(fpr, tpr, 'x');

end

