%%

% We will assume our topic model has T topics, N users, F features,
% Ni action-sequences for the i-th user

% IMPORTANT: We assume that the input feature vectors are sorted by userids

T = 3;
U = unique(fvs(:,1));
N = length(U); % No. users
F = size(fvs,2)-1; % The first column is the userid

Ni = zeros(size(U,1),1);
for i=1:N
    Ni(i) = length(find(fvs(:,1)==U(i)));
end

% The two Dirichlet priors
alpha = 1;
Alpha = alpha*ones(T,1);

beta  = 1;
Beta = beta*ones(F,1);

AlphaSum = sum(Alpha);
BetaSum = sum(Beta);

% Initialize the Theta's. These are action-type distributions
Theta = zeros(T,N);
for i=1:N
    Theta(:,i) = drchrnd(Alpha',1)';
end

% Initialize the Tau's. These are distributions over actions for each topic
Tau = zeros(F,T);
for k=1:T
    Tau(:,k) = drchrnd(Beta',1)';
end

% Initialize Z's to random
Z = zeros(size(fvs,1),1);
j = 1;
for i=1:N
    MNRnd = mnrnd(1,Theta(:,i),Ni(i));
    Z(j:j+Ni(i)-1) = arrayfun(@(x) find(MNRnd(x,:)==1), 1:size(MNRnd,1));
    j = j+Ni(i);
end
clear MNRnd

% Precompute number of words in each topic
Nk = zeros(F,T);
for k=1:T
    fvs_k = fvs(Z==k,2:end);
    Nk(:,k) = sum(fvs_k,1)';
end
clear fvs_k

Nk_Rsum = sum(Nk);

% Precompute number of sequences in each topic for each user
Nn = zeros(T,N);
j = 1;
for i=1:N
    Nn(:,i) = histc(Z(j:j+Ni(i)-1),1:T);
    j = j+Ni(i);
end

% Precompute number of actions in each document.
% Assume that each action is coded as boolean (1|0).
Nw = sum(fvs(:,2:end),2);

TauHist = [];
ThetaHist = [];

% Start the Collapsed Gibbs Sampling step

rng('default');
%rand('seed',43); % Octave

% Initialize all indexes having non-zero values
len = size(fvs,1);
M = cell(len,1);
SV = cell(len,1);
for i=1:len
    fv = fvs(i,2:end);
    M{i,1} = find(fv>0);
    SV{i,1} = fv(:,M{i,1})'; % Sparse Vector
end

for epoch=1:200

    % Assume i = (n,m) where n=user, m=action sequence
    % Compute p(zi = k | Z_(-i), W)
    p = 1;
    for i=1:N
        for j=1:Ni(i) % Each document of i-th user
            
            idx = p+j-1;
            %fv = fvs(idx,2:end);  % curr doc -- j-th document of i-th user
            z = Z(idx);           % label of document
            l = Nw(idx);          % Number of words in curr document
            m = M{idx,1}; %find(fv>0);
            sv = SV{idx,1};
            Nk_Rsum(z) = Nk_Rsum(z) - l;

            % Adjusted document counts for k-th topic excluding curr doc
            if Nn(z,i) == 0 
                fprintf('Found 0 for z=%d, i=%d, j=%d\n',z,i,j);
            end
            Nn(z,i) = Nn(z,i) - 1;

            % Adjusted word counts for k-th topic excluding curr doc
            %Nk(:,z) = Nk(:,z) - fv';
            Nk(m,z) = Nk(m,z) - sv;
            
            % Numerator of p(zi = k | Z_(-i), W)
            P = zeros(1,T);
            for k=1:T
                %P(k) = prod( ...
                %        (Nk(m,k)+Beta(m))' ./ ((1:l)+(sum(Nk(:,k)+Beta)-1)) ...
                %    ) * (Nn(k,i)+Alpha(k));
                %P(k) = prod( ...
                %        (Nk(m,k)+Beta(m))' ./ ((1:l)+(Nk_Rsum(k)+BetaSum-1)) ...
                %    ) * (Nn(k,i)+Alpha(k));
                P(k) = 1;
                ll = 0;
                for mm=1:size(m,2)
                    for ssv=1:sv(mm);
                        P(k) = P(k)*(Nk(m(mm),k)+Beta(m(mm))+ssv-1)/(Nk_Rsum(k)+BetaSum+ll);
                        ll = ll+1;
                    end
                end
            end
            P = P / sum(P);

            % sample new z
            nz = find(mnrnd(1,P,1)==1);

            % Readjusted word counts for k-th topic including curr doc
            %Nk(:,nz) = Nk(:,nz) + fv';
            Nk(m,nz) = Nk(m,nz) + sv;
            Nk_Rsum(nz) = Nk_Rsum(nz) + l;

            % Readjusted document counts for k-th topic including curr doc
            Nn(nz,i) = Nn(nz,i) + 1;

            % Change the label for curr doc
            Z(idx) = nz;

        end
        fprintf('Completed Gibbs Sampling Epoch %d User %d...\n',epoch,i);
        p = p+Ni(i);
    end
    
    for k=1:T
        Tau(:,k) = (Nk(:,k)+Beta) ./ sum(Nk(:,k)+Beta);
    end
    for i=1:N
        Theta(:,i) = (Nn(:,i)+Alpha) ./ sum(Nn(:,i)+Alpha);
    end
    
    TauHist = [TauHist; Tau];
    ThetaHist = [ThetaHist; Theta];
    %disp(Theta);
    
    fprintf('Completed Gibbs Sampling Epoch %d...\n',epoch);

end

%%

save(strcat(outpath,'/inferred-params.mat'), 'TauHist', 'ThetaHist');
clear TauHist ThetaHist

%%

load(strcat(outpath,'/inferred-params.mat'));

%%

% Analyze Convergence
A = ThetaHist(:,1); % Convergence of one parameter
s = size(A,1)/3-2;
plot(0:s,A((0:s)*3+1));
