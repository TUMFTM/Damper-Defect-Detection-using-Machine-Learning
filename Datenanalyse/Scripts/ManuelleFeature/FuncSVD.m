function FuncSVD = FuncSVD(data,opts)
% Singular value decomposition of measurement blocks

%% SVD

fieldsData = fields(data);

numSens = size(fieldsData,1);
numObs = size(data.(fieldsData{1}),1);
FuncSVD.raw = zeros(numObs,numSens);
FuncSVD.norm = zeros(numObs,numSens);
for cntObs = 1 : numObs
    A = zeros(size(data.(fieldsData{1}),2),numSens);
    for cntSens = 1 : numSens
        A(:,cntSens) = data.(fieldsData{cntSens})(cntObs,:)';
    end
    [~,S,~] = svd(A,'econ');
    FuncSVD.raw(cntObs,:) = diag(S);
    FuncSVD.norm(cntObs,:) = FuncSVD.raw(cntObs,:)./S(1,1);
end

end