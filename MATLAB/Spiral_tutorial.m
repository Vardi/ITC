
%% Tutorial for KNN ITC

% Set up environment variabels

addpath('../Data')
set(0,'DefaultFigureWindowStyle','docked')
set(0,'defaulttextinterpreter','latex','DefaultAxesFontSize',20)

%% Load data

Spiral = dlmread('Spiral.txt');
y = Spiral(:,3); %Class labels
Spiral(:,3) = [];

[N d] = size(Spiral);

%% Normalize data

%Normalize range to -1, 1
Spiral = Spiral - repmat(min(Spiral),N,1);
Spiral = Spiral ./ repmat(max(Spiral),N,1);
Spiral = 2*Spiral;
Spiral = Spiral - ones(N,d);

%% Run clustering algorithm

%Setup parameters
K0      = 12;           % Number of starting clusters
N0      = 3;            % Number of points to include in starting cluster
method  = 'knn';        % Method
var1    = 1;            % Method parameter
endc    = 2;            % Number of end clusters
a       = 1;            % Alpha parameter
visualize = 1;          % Visualization flag

parameterString = fprintf('K0 = %d, N0 = %d k=%d  endc=%d  a=%d', ...
                            K0, N0, var1, endc, a);

clear SpiralData;
SpiralData = ITCDataHandler('points',Spiral);

C = SpiralData.ReduceClusters(  'visualize', visualize,  ...
                                'var1', var1,            ...
                                'K0', K0,                ...
                                'N0', N0,                ...
                                'endcluster', endc,      ...
                                'a',a);


%% Routine for flipping labels around

% Labels from result with 3 cluseters
labels = C(K0-endc+1).labels;
cluster_names = unique(C(K0-endc+1).labels);

maxAcc = 0;
maxInd = 1;

perm = perms(unique(y));
[pN pM] = size(perm);

true_labels = labels;

for i=1:pN
    flipped_labels = zeros(1,N);
    for cl = 1 : pM
        flipped_labels(labels==cluster_names(cl)) = perm(i,cl);
    end
    
    testAcc = sum(flipped_labels == y')/N;
    if testAcc > maxAcc
        maxAcc = testAcc;
        maxInd = i;
        true_labels = flipped_labels;
    end
    
end

disp(['Accuracy found to be: ' num2str(maxAcc) '.']);

%% Create confusion matrix
CM = zeros(pM,pM);
for rc = 1 : pM
    for cc = 1 : pM
        CM(rc,cc) = sum( ((y'==rc) .* (true_labels==cc)) );
    end
end






