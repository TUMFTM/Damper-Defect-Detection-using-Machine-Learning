function m = multiClassAUC(pClass, classLabel)
            %multiClassAUC - Compute Hand & Till's M measure (the area under the ROC for two-class classifiers)
            %   The receiver operating characteristic curve (ROC) is frequently used to characterize the
            %   capability of a two-class classifier. In particular, the area under the ROC curve (AUC) is equal
            %   to the probability that a classifier designed to detect a first class will assign a higher
            %   probability of membership to a randomly chosen member of that class than to a randomly chosen
            %   member of the alternative class. This makes the AUC a particularly useful measure of the
            %   effectiveness of a two-class classifier. The value of the AUC can range from 0.0 (classification
            %   is 100% incorrect) to 1.0 (100% correct), with a value of 0.5 equal to the performance of a
            %   random classifier. The AUC is particularly useful for feature selection in a multi-feature
            %   classifier to determine if the addition of a feature (forward selection) or deletion of a
            %   feature (backward elimination) improves the performance of the classifier.
            %
            %   The AUC is most commonly computed by generating the ROC curve and numerically integrating it.
            %   Unfortunately, this technique is subject to inaccuracies and bias related to the methods used
            %   to generate the ROC curve and perform the numerical integration. It also becomes challenging for
            %   multi-class problems since the AUC for a c-class classifier generalizes to the volume under the
            %   ROC surface of a c-dimensional hypercube and requires c-dimensional integration.
            %
            %   Hand & Till have developed an alternative method based on the probabilistic definition of the
            %   AUC and dubbed it the M measure (see description above, and References below). The M measure is
            %   a rank order calculation related to the Mann-Whitney-Wilcoxon statistic. It does not require the
            %   creation or numerical integration of an ROC curve/surface and is therefore not subject to
            %   inaccuracies and biases associated with those techniques. In this sense it is an "exact"
            %   method, and generalizes to the multi-class problem without modification.
            %
            %   Usage:
            %       m = multiClassAUC(pClass, classLabel)
            %
            %   Inputs:
            %       pClass          an n-by-c table or matrix of class membership "probabilities". These are
            %                       "probabilites" in the sense that each value must lie within the closed
            %                       interval [0, 1], larger values correspond to a greater likelihood of class
            %                       membership, and sum(pClass(i, :)) must equal unity. pClass(i, j) is the
            %                       probability that observation i belongs to class j.
            %
            %       classLabel      an n-by-1 vector or table of class labels, wherein classLabel(i) is the
            %                       known class of the ith observation. The labels must be integers but are not
            %                       constrained to positive or consecutive values. The labels must correspond to
            %                       the columns of pClass in ordinal order, e.g., the lowest value corresponds
            %                       to column 1 and the highest to column c.
            %
            %   Outputs:
            %       m               Hand & Till's M measure (generalized AUC). Output values lie within the
            %                       closed interval [0, 1].
            %
            %   References:         Hand DJ, Till RJ, "A Simple Generalization of the Area Under the ROC Curve
            %                       for Multiple Class Classification Problems, Machine Learning, 45, 171-186,
            %                       2001.
            
            % from https://de.mathworks.com/matlabcentral/fileexchange/71158-multiclassauc?s_tid=LandingPageTabfx&s_tid=mwa_osa_a
            
            % parse inputs
            p = inputParser;
            p.addRequired('pClass', @(x) (isnumeric(x) && ismatrix(x)) || (istable(x) && size(x, 2) > 1));
            p.addRequired('classLabel', @(x) (isnumeric(x) || istable(x)) && iscolumn(x));
            p.parse(pClass, classLabel);
            pClass = p.Results.pClass;
            classLabel = p.Results.classLabel;
            assert(size(pClass, 1) == size(classLabel, 1));
            
            % convert tables to arrays
            if istable(pClass)
                pClass = table2array(pClass);
                assert(isnumeric(pClass));
            end
            if istable(classLabel)
                classLabel = table2array(classLabel);
                assert(isnumeric(classLabel));
            end
            
            % capture the set of class labels
            classLabelSet = unique(classLabel);
            classCount = length(classLabelSet);
            
            % class sample counts
            classSampleCount = zeros(1, length(classLabelSet));
            for ix = 1:classCount
                classSampleCount(ix) = sum(classLabel == classLabelSet(ix));
            end
            
            % for each class, rank order the samples by probability of membership
            [~, ixSort] = sort(pClass);
            pSortedLabel = zeros(size(ixSort));
            for ix = 1:classCount
                pSortedLabel(:, ix) = classLabel(ixSort(:, ix));
            end
            
            % compute Hand & Till's M measure (identical to AUC for two classes)
            Asum = 0;
            for j =1:(classCount  - 1)
                jLabel = classLabelSet(j);
                for k = (j+1):classCount
                    kLabel = classLabelSet(k);
                    
                    % class counts
                    nj = classSampleCount(j);
                    nk = classSampleCount(k);

                    % compute A(j|k)
                    thisSortedLabel = pSortedLabel(pSortedLabel(:, j) == jLabel | pSortedLabel(:, j) == kLabel, j);
                    sumRank = sum(find(thisSortedLabel == jLabel));
                    Ajk = (sumRank - nj * (nj + 1) / 2) / (nj * nk);
                    
                    % compute A(k|j)
                    thisSortedLabel = pSortedLabel(pSortedLabel(:, k) == jLabel | pSortedLabel(:, k) == kLabel, k);
                    sumRank = sum(find(thisSortedLabel == kLabel));
                    Akj = (sumRank - nk * (nk + 1) / 2) / (nj * nk);
                    
                    Asum = Asum + (Ajk + Akj) / 2;
                end
            end
            m = 2 * Asum / (classCount * (classCount - 1));
        end
