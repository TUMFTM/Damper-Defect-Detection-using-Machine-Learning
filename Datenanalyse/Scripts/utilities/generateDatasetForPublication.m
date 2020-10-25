function [data] = generateDatasetForPublication()
%GENERATEDATASETFORPUBLICATION Summary of this function goes here
%   Detailed explanation goes here

    filename = 'Label_DD2.xlsx';
    foldername = 'W:\Projekte\Fahrwerkdiagnose\Datenanalyse\Datensatz\';
    opts = setOptions();

    data = loadData(filename, foldername, opts, 'publication', 1);
end

