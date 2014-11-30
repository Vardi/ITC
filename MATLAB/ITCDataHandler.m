classdef ITCDataHandler < handle
%
% Class for handling datasets and performing information theoretic
% clustering using a heuristic optimization schema and a knn divergence
% measure.
%
%
% Contributors: Vidar Vikjord and Robert Jenssen.
%
    
    properties
        points
        labels
        classes
        numbClasses
        numbPoints
        dim
        CM
        R
    end

    methods

        function obj = ITCDataHandler(varargin)
        % Constructor for ITCDataHandler object.

            % Set defaults and parse input
            [~, prop] = parseparams(varargin);
            obj.labels = NaN;
            obj.points = NaN;
            obj.dim = NaN;
            obj.numbPoints = NaN;
            obj.R = NaN;
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'labels')
                        obj.labels = prop{i+1};
                    elseif strcmpi(prop{i},'points')
                        obj.points = prop{i+1};
                    elseif strcmpi(prop{i},'dim')
                        obj.dim = prop{i+1};
                    elseif strcmpi(prop{i},'numbPoints')
                        obj.numbPoints = prop{i+1};
                    end
                end
            end

            if isnan(obj.numbPoints);
            obj.numbPoints = length(obj.points);end
            if isnan(obj.dim)
            obj.dim = min(size(obj.points));end;

            obj.classes = unique(obj.labels);obj.classes(obj.classes == 0) = [];
            obj.numbClasses = length(obj.classes);
            obj.CM = [0 0 0; jet(obj.numbClasses)];

             if isnan(obj.labels)
                obj.labels = zeros(1,obj.numbPoints);
             end

             obj.R = dist(obj.points');

        end

        function fig = Visualize(obj,fig)
            if nargin == 1
                fig = figure();
            end
            if ~ishandle(fig)
                warning('ITCDataHandler:InputWarning','Input is not a figure handle. Creating new figure.');
                fig = figure();
            end
            if obj.dim ~= 2
                error('Can not visualize data not 2-dimensional');
            end

            figure(fig);hold on
            for i=0:obj.numbClasses
                if i==0
                    plot(obj.points(obj.labels == i,1),...
                        obj.points(obj.labels == i,2),'.','Color',obj.CM(i+1,:));
                elseif not(isnan(obj.classes))
                    plot(obj.points(obj.labels == obj.classes(i),1),...
                        obj.points(obj.labels == obj.classes(i),2),'.','Color',obj.CM(obj.classes(i)+1,:));
                    text(obj.points(obj.labels == obj.classes(i),1),...
                        obj.points(obj.labels == obj.classes(i),2),num2str(obj.classes(i)));
                end
            end

            hold off

        end

        function Ent = Calculate_Entropy(obj, varargin)

            % Input parsing
            [~, prop] = parseparams(varargin);
            var1 = 1;
            report = 0;
            method = 'knn';
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'var1')
                        var1 = prop{i+1};
                    elseif strcmpi(prop{i},'report')
                        report = prop{i+1};
                    elseif strcmpi(prop{i},'method')
                        method = prop{i+1};
                    end
                end
            end

            % Calculate K Nearest Neightbor Entropy estimate
            if strcmp(method,'knn')
                k = var1;

                n = obj.numbPoints;
                [Rs, idx]=sort(obj.R);

                V = zeros(n,n);
                for i=1:n
                   V(idx(k,i),i) = 1/(ITCDataHandler.hypervolume(Rs(k,i),obj.dim));
                end

                Ent =-log10(k*sum(V(:))/n^2); %?!?! Removed n^2

            % Calculate Parzen windowing Entropy estimate
            elseif strcmp(method,'parzen')
                sig = var1;

                n = length(obj.points);
                kernel = @(r) 1/((2*pi)^(obj.dim/2)*det(sig*eye(obj.dim)))*exp(-(r^2)/(2*sig^2));

                K = zeros(n,n);
                for i=1:n
                    for j=1:n
                        K(i,j) = kernel(obj.R(i,j));
                    end
                end

                Ent = -log10(1/n^2 * sum(K(:)));

            end

            if report;
                disp(['Entropy using ' method ' with ' ...
                'variable ' num2str(var1) ' was: ' num2str(Ent)]);
            end

        end

        function [fig PPDF1 XX YY] = EstimatePDF(obj, varargin)

            if obj.dim > 2;error('Can not construct pdf-estimate for high dimensional data.');end

            % Input parsing
            method = 'parzen';
            var1   = sqrt(mean(var(obj.points)));
            Border = var1;
            stepSize = 0.5;
            fig = NaN;
            normalize = 0;

            [~, prop] = parseparams(varargin);
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'var1')
                        var1 = prop{i+1};
                    elseif strcmpi(prop{i},'method')
                        method = prop{i+1};
                    elseif strcmpi(prop{i},'border')
                        Border = prop{i+1};
                    elseif sum(strcmpi(prop{i},{'fig','figure'}))
                        fig = prop{i+1};
                        figure(fig);
                    elseif strcmpi(prop{i},'normalize')
                        normalize = prop{i+1};
                    elseif strcmpi(prop{i},'stepSize')
                        stepSize = prop{i+1};
                    end
                end
            end

            Xrange = [min(obj.points(:,1))-Border max(obj.points(:,1))+Border];
            Yrange = [min(obj.points(:,2))-Border max(obj.points(:,2))+Border];

            if not(ishandle(fig));fig = figure();end

            % Setup coordinate grid
            [XX YY] = meshgrid(Xrange(1):stepSize:Xrange(2), Yrange(1):stepSize:Yrange(2));
            YY = flipud(YY);
            PPDF1 = zeros(size(XX));

            if strcmp(method,'parzen')

                % Parzen function handle
                pf1 = @(C1,C2) (1/obj.numbPoints)*(1/((2*pi)*var1^2)).*...
                         exp(-( (C1(1)-C2(1))^2+ (C1(2)-C2(2))^2)/(2*var1^2));


                % Populate coordinate surface
                [RR CC] = size(PPDF1);
                for c=1:CC
                   for r=1:RR
                       for d=1:obj.numbPoints
                            PPDF1(r,c) = PPDF1(r,c) + ...
                                pf1([XX(1,c) YY(r,1)],[obj.points(d,1) obj.points(d,2)]);
                       end
                   end
                end

            elseif strcmp(method,'knn')

                % Number of neighbors to find
                k=ceil(var1);

                % Dimension of data
                d = obj.dim;

                % Set up space to estimate
                PPDF1 = zeros(size(XX));
                [RR CC] = size(PPDF1);

                % Populate coordinate surface
                for c=1:CC
                   for r=1:RR
                       [~, dist] = knnsearch(obj.points,[XX(1,c) YY(r,1)],'K',k);
                       PPDF1(r,c) = (k/obj.numbPoints)*(1/(dist(end))^d);
                   end
                end

            end

            % Normalize data
            if normalize
                m3 = max(PPDF1(:));
                PPDF1 = PPDF1 / m3;
            end

            % Visualize results
            % Set up visualization
            stem3(obj.points(:,1),obj.points(:,2),zeros(obj.numbPoints,1),'b.');
            hold on;
            % Add PDF estimates to figure
            s3 = surfc(XX,YY,PPDF1);shading interp;alpha(s3,'color');
        end

        function K = CalculateKernelMatrix(obj, varargin)
        % Calculate Kernel matrix for dataset using either
        % a gaussian kernel or the k nearest neighbors.
        % Input:
        %    VARARGIN:
        %        'method' = 'parzen'
        %            Either 'parzen' or 'knn'
        %        'var1' = 0.5
        %            Parameter for the choosen method. Std of
        %            gaussian for 'parzen' and which
        %            neightborsing point to evaluate for 'knn'
             
            % Input parsing
            [~, prop] = parseparams(varargin);
            var1 = 0.5;
            method = 'parzen';
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'var1')
                        var1 = prop{i+1};
                    elseif strcmpi(prop{i},'method')
                        method = prop{i+1};
                    end
                end
            end

            switch method
                case 'parzen'

                    S = var1^2*eye(obj.dim);
                    detS = det(S);

                    kernel = @(r) 1/...
                                    ((2*pi)^(obj.dim/2)*sqrt(detS))*...
                                    exp(-0.5*(r.^2./var1^2));

                    K = zeros(obj.numbPoints);
                    for i=1:obj.numbPoints
                        for j=1:obj.numbPoints
                            K(i,j) = kernel(obj.R(i,j));
                        end
                    end

                case 'knn'
                    error('Not yet implementet');
            end
        end

        function CSDivInfo = FindWorstCluster(obj, varargin)
           % Function for calculating divergence between data
           % clusters

           [~, prop] = parseparams(varargin);
           var1 = 5;
           a    = 0.2;
           method = 'knn';
           for i=1:length(prop)
                if strcmpi(prop{i},'var1')
                    var1 = prop{i+1};
                elseif strcmpi(prop{i},'a')
                    a = prop{i+1};
                elseif strcmpi(prop{i},'method')
                    method = prop{i+1};
                end
           end

           [Rs idx] = sort(obj.R);
           Vs = ITCDataHandler.hypervolume(Rs,obj.dim);

           % Assign to cluster yielding highest cs-divergence
           CSDivInfo = [1 -inf]; % ['Class for higests cs div', 'CS Divergence']
           labelsBackup = obj.labels;
           classesBackup = obj.classes;

            for t = 1: obj.numbClasses

                % Remove one cluster
                obj.labels = labelsBackup;
                obj.classes = classesBackup;
                classRemoved = obj.classes(t);
                obj.labels( obj.labels==obj.classes(t) ) = 0;
                obj.classes(obj.classes==obj.classes(t)) = [];

                % Matrix for all cross divergence terms
                M = zeros(obj.numbClasses-1,obj.numbClasses-1);

                switch method

                    case 'knn'

                    for ri = 1 : obj.numbClasses-1

                        %Extract relevant info from hypervolume matrix
                        VsE = Vs(:,(obj.labels==obj.classes(ri)));

                        for ci = 1 : obj.numbClasses-1

                            idxE = idx(:,obj.labels==obj.classes(ri));
                            idxE( obj.labels(idxE) ~= obj.classes(ci)) = 0;

                            [r, c, ~] = find(idxE~=0);

                            % Number of points in each class
                            NO = size(VsE,2);       %Outer class number of points
                            %NI = length(r(c==1));   %Inner class number of points

                            for ni = 1 : NO
                                p = r(c==ni);        % Index of nearest neightbor from outer to inner

                                hyperV = VsE(p(var1+(obj.classes(ri)==obj.classes(ci))),ni);

                                if hyperV <= eps;continue;end
                                M(ri,ci) = M(ri,ci) + 1/hyperV^a;

                            end
                        end
                    end

                    case 'parzen'

                        for ri = 1 : obj.numbClasses-1
                            for ci = 1 : obj.numbClasses-1

                                RE = obj.R(obj.labels==obj.classes(ri),obj.labels==obj.classes(ci));
                                KerV = CalcKernel_vec(RE(:),var1,obj.dim);
                                M(ri,ci) = mean(KerV(:));
                            end
                        end

                end

                % Sum up divergence matrix and compare
                M = 0.5*(M' + M);
                CS = 0;
                for ri = 1:obj.numbClasses-2
                    for ci = ri+1:obj.numbClasses-1
                        CS = CS + M(ri,ci)/sqrt(M(ri,ri)*M(ci,ci));
                    end
                end

                CS = -log(CS);

                if CS > CSDivInfo(2)
                    CSDivInfo = [classRemoved,CS];
                end

            end

            obj.labels = labelsBackup;
            obj.classes = classesBackup;

        end

        function obj = SeedClusters(obj, varargin)
        % Function for seeding initial clusters
            
            if sum(obj.labels) > 0;
                warning('ITCDataHandler:GeneralWarning',...
                    'Previous label information deleted.')
            end

           [~, prop] = parseparams(varargin);
           K0 = 5;
           N0 = 5;
           for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'K0')
                        K0 = prop{i+1};
                    elseif strcmpi(prop{i},'N0')
                        N0 = prop{i+1};
                    end
                end
           end

           % Recreate colormap
           obj.CM = [0 0 0 ; jet(K0)];

           [Rs idx] = sort(obj.R);

           N = obj.numbPoints;
           Clusters = zeros(1,N);
           counter = 1:N;

           for kInd=1:K0

                % Vector to store points found
                Points_added = zeros(1,N0);

                % Select random point not already clustered
                Potential_points = counter((Clusters(1,:) == 0));
                Point_choosen = Potential_points(randi(length(Potential_points),1));
                Points_added(1) = Point_choosen;

                % Store label
                Clusters(1,Point_choosen) = kInd;

                % Iterativly add N0-1 nearest neighbors to cluster to make a
                % cluster of N0 points
                for nInd = 2 : N0
                    % Find candidates based on points currently in cluster
                    Candidates = idx(:,Points_added(1:nInd-1));           % Optimalization by only extracting some points possible
                    Candidates_distance = Rs(:,Points_added(1:nInd-1));
                    % Find closest one not currently in cluster
                    done = 0;
                    while not(done)
                        % Find lowest distance
                        [n, m] = find(Candidates_distance==min(Candidates_distance(:)));

                        % Iterate over n,m if more than one point if found
                        for i = 1:length(n)
                            % Check if point is already added to a cluster
                            if Clusters(1,Candidates(n(i),m(i))) == 0
                               done = 1;
                               Clusters(1,Candidates(n(i),m(i))) = kInd;
                               Points_added(nInd) = Candidates(n(i),m(i));
                               break;
                            else
                               Candidates_distance(n(i),m(i)) = inf;
                            end
                        end
                    end
                end
           end

           obj.labels = Clusters;
           obj.classes = unique(obj.labels); obj.classes(obj.classes==0)=[];
           obj.numbClasses = K0;
        end

        function obj = ClusterUnlabeled(obj,varargin)
        % Cluster all datapoints not currently assigned to a cluster
        
           % Input parsing
           [~, prop] = parseparams(varargin);
           var1    = 1;
           a       = 1;
           visualize = 0;
           debug = 0;
           for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'var1')
                        var1 = prop{i+1};
                    elseif strcmpi(prop{i},'visualize')
                        visualize = prop{i+1};
                    elseif strcmpi(prop{i},'a')
                        a = prop{i+1};
                    elseif strcmpi(prop{i},'debug')
                        debug = prop{i+1};
                    end
                end
           end

            % Calculate distances, sort and apply hypervolume
            [Rs idx] = sort(obj.R);
            Vs = ITCDataHandler.hypervolume(Rs,obj.dim);

            % Calculate Cluster Prototypes
            CP = struct('mean',[0 0],'cov',eye(2));
            for i=1:obj.numbClasses
                CP(i).mean = mean( obj.points(obj.labels(1,:)==i,:),1);
                CP(i).cov = cov( obj.points(obj.labels(1,:)==i,:) );
            end

            % Set up visualization
            if visualize && obj.dim == 2
                f = figure();hold on;
                for k=0:obj.numbClasses
                    plot(obj.points(obj.labels==k,1),obj.points(obj.labels==k,2),...
                        '.','Color',obj.CM(k+1,:))
                    if k>0;text(obj.points(obj.labels == obj.classes(k),1),...
                        obj.points(obj.labels == obj.classes(k),2),num2str(obj.classes(k)));end
                end
            end

            % Find index of unlabeled points
            Points_to_cluster = find(obj.labels ==0 );
            Pcounter = 1:obj.numbPoints;

            % Do until all is clustered
            while not(isempty(Points_to_cluster))

                % Find point closest to a cluster mean or another clustered point
                clustered_indexes = obj.labels ~= 0;

                t = obj.R(Points_to_cluster,:);
                tt = t(:,clustered_indexes);
                [p k d] = find(tt==min(tt(:)));

                % Vector to keep track of closest point
                Closest = [d p k]; %[distance , pointind (relative to 'Points_to_cluster'), cluster]

                % Keep track of which assignment gave highest divergence
                CSDivInfo = [1 -inf]; % ['Class for higest cs div', 'CS Divergence']
                labelsBackup = obj.labels;

                % Assign to cluster yielding highest cs-divergence
                for t = 1: obj.numbClasses

                    % Assign point to class for testing
                    obj.labels = labelsBackup;
                    obj.labels( Points_to_cluster(Closest(2)) ) = obj.classes(t);

                    % Matrix for all cross divergence terms
                    M = zeros(obj.numbClasses,obj.numbClasses);

                    for ri = 1 : obj.numbClasses

                        % Extract relevant info from hypervolume matrix
                        VsE = Vs(:,Pcounter(obj.labels==obj.classes(ri)));

                        for ci = 1 : obj.numbClasses

                            idxE = idx(:,obj.labels==obj.classes(ri));
                            idxE( obj.labels(idxE) ~= obj.classes(ci)) = 0;

                            [r, c, ~] = find(idxE~=0);

                            % Number of points in each class
                            NO = size(VsE,2);
                            %NI = length(r(c==1));

                            for ni = 1 : NO
                               p = r(c==ni); % This is generally the most expensive step because it gets called so many times.

                               if ci==ri
                                  hyperV = VsE(p(end),ni);  % Do complete link if evaluating against self
                               else
                                  hyperV = VsE(p(var1+(obj.classes(ri)==obj.classes(ci))),ni); % Ignore overlapping neighbors
                               end

                               M(ri,ci) = M(ri,ci) + 1/(hyperV)^(a);
                            end

                        end
                    end

                    % Sum up divergence matrix and compare
                    M = 0.5*(M' + M);
                    CS = 0;
                    for ri = 1:obj.numbClasses-1
                        for ci = ri+1:obj.numbClasses
                            CS = CS + M(ri,ci)/sqrt(M(ri,ri)*M(ci,ci));
                        end
                    end
                    CS = -CS;

                    if debug
                        disp('M matrix: \n')
                        disp(M)
                        disp('Cost function: \n')
                        disp(CS)
                    end

                    if CS > CSDivInfo(2)
                        CSDivInfo = [obj.classes(t) , CS];
                    end
                end

                % Assign to closest cluster and update clusters and remove point
                obj.labels(Points_to_cluster(Closest(2))) = CSDivInfo(1);
                CP(CSDivInfo(1)).mean = mean(obj.points(labelsBackup==CSDivInfo(1),:));
                CP(CSDivInfo(1)).cov  =  cov(obj.points(labelsBackup==CSDivInfo(1),:));

                % Visualize if flag set
                if visualize && obj.dim == 2
                    plot( obj.points( Points_to_cluster(Closest(2)),1 ), ...
                          obj.points( Points_to_cluster(Closest(2)),2 ), ...
                          '.','Color',obj.CM(CSDivInfo(1)+1,:))
                      text( obj.points(Points_to_cluster(Closest(2)),1),...
                            obj.points(Points_to_cluster(Closest(2)),2), ...
                            num2str(obj.labels(Points_to_cluster(Closest(2)))));
                      drawnow
                end

                if debug
                    disp(['Point ' num2str(Points_to_cluster(Closest(2))) ' assigned to ' num2str(CSDivInfo(1)) '.']);
                    pause
                end

                % Remove clustered point
                Points_to_cluster(Closest(2)) = [];

            end

            % Close visualization
            if visualize && obj.dim==2;close(f);end

        end

        function C = ReduceClusters(obj, varargin)
        % Function controlling main ITC algorithm reducing clusters

            % Input parsing
            var1            = 5;
            a               = 0.2;
            endclusters     = 2;
            K0              = 10;
            N0              = 8;
            visualize       = 0;
            method          = 'knn';

            [~, prop] = parseparams(varargin);
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'var1')
                        var1 = prop{i+1};
                    elseif strcmpi(prop{i},'a')
                        a = prop{i+1};
                    elseif strcmpi(prop{i},'endcluster')
                        endclusters = prop{i+1};
                    elseif strcmpi(prop{i},'K0')
                        K0 = prop{i+1};
                    elseif strcmpi(prop{i},'N0')
                        N0 = prop{i+1};
                    elseif strcmpi(prop{i},'visualize')
                        visualize = prop{i+1};
                    elseif strcmpi(prop{i},'method')
                        method = prop{i+1};
                    end
                end
            end

            C = struct('labels',0,'M',0,'CS',0);

            obj.SeedClusters('K0',K0,'N0',N0);

            [CS M] = obj.CalculateDivergence('var1',var1,'a',a);
            C(1).labels = obj.labels;
            C(1).M = M;
            C(1).CS = CS;

            obj.ClusterUnlabeled('var1',var1,'method',method,'visualize',visualize,'a',a);
            [CS M] = obj.CalculateDivergence('var1',var1,'a',a);
            C(2).labels = obj.labels;
            C(2).M = M;
            C(2).CS = CS;

            for cc = 3 : K0 - endclusters + 2

                T = obj.FindWorstCluster('var1',var1,'a',a);
                obj.ShatterCluster('remove',T(1));

                obj.ClusterUnlabeled('var1',var1,'method',method,'visualize',visualize,'a',a);
                [CS M] = obj.CalculateDivergence('var1',var1,'a',a);

                C(cc).labels = obj.labels;
                C(cc).M = M;
            	C(cc).CS = CS;

                if visualize;
                    obj.Visualize;
                    title(['Divergence: ' num2str(CS) '.']);
                end

                disp(['Current cluster number: ' num2str(K0-cc+2)])

            end
        end

        function obj = ShatterCluster(obj, varargin)
        % Function for removing one cluster from dataset.
        
           remove = [];
           [~, prop] = parseparams(varargin);

           for i=1:length(prop)
               if strcmpi(prop{i},'remove')
                   remove = prop{i+1};
               end
           end

           for i = 1 : length(remove)
              if nnz(obj.labels==remove(i)) > 0
                  obj.labels(obj.labels==remove(i)) = 0;
                  obj.classes(obj.classes==remove(i)) = [];
                  obj.numbClasses = obj.numbClasses - 1;
              else
                  warning('ITCDataHandler:clusterError',...
                      ['Cluster ' num2str(remove(i)) ' not found.'])
              end
           end
        end

        function [CS M] = CalculateDivergence(obj, varargin)
        % Calculate divergence for given object state.
        
            % Input parsing
            [~, prop] = parseparams(varargin);
            a       = 1;
            debug   = 0;
            for i=1:length(prop)
                if ischar(prop{i})
                    if strcmpi(prop{i},'a')
                        a = prop{i+1};
                    elseif strcmpi(prop{i},'debug')
                        debug = prop{i+1};
                    end
                end
            end

            if obj.numbClasses == 1; CS = 0; M = 0; return;end

            [Rs idx] = sort(obj.R);
            Vs = ITCDataHandler.hypervolume(Rs, obj.dim);

            % Matrix for all cross divergence terms
            M = zeros(obj.numbClasses,obj.numbClasses);

            for ri = 1 : obj.numbClasses

                % Extract relevant info from hypervolume matrix
                VsE = Vs(:,(obj.labels==obj.classes(ri)));

                for ci = 1 : obj.numbClasses

                    idxE = idx(:,obj.labels==obj.classes(ri));
                    idxE( obj.labels(idxE) ~= obj.classes(ci)) = 0;

                    [r, c, ~] = find(idxE~=0);

                    % Number of points in each class
                    NO = size(VsE,2);

                    for ni = 1 : NO
                        p = r(c==ni);
                        hyperV = mean(VsE(p(1:end),ni));

                        if hyperV <= eps;continue;end

                        M(ri,ci) = M(ri,ci) + 1/hyperV^a;
                    end

                end
            end

            % Sum up divergence matrix and compare
            M = 0.5*(M' + M);
            CS = 0;
            C = nchoosek(obj.numbClasses,min(2,obj.numbClasses));
            for ri = 1:obj.numbClasses-1
                for ci = ri+1:obj.numbClasses
                    CS = CS + M(ri,ci)/(C*sqrt(M(ri,ri)*M(ci,ci)));
                end
            end

            CS = -log(CS);

            if debug;
                disp(M)
            end
        end
        
    end
    
    methods(Static)
        function V = hypervolume(r,d)
        % Calculate volume of hypersphere with radius 'r' in 'd' 
        % dimensions

            V = pi^(d/2)/(gamma(d/2+1)).*r.^d;
        end
    end
end

