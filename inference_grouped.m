clear all

%%

% We will assume our topic model has T topics, N users, F features, K
% groups, and Ni action-sequences for the i-th user

% IMPORTANT: We assume that the input feature vectors are sorted by userids

%pkg load 'statistics' % Octave

rng('default'); % MATLAB
%rand('seed',43); % Octave

T = 3;
K = 3;
U = sort(unique(fvs(:,1)));
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

eta  = 1;
Eta = eta*ones(K,1);

AlphaSum = sum(Alpha);
BetaSum = sum(Beta);

% Initialize the Theta's. These are action-type distributions
Theta = zeros(T,K);
for i=1:K
    Theta(:,i) = drchrnd(Alpha',1)';
end

% Initialize the Tau's. These are distributions over actions for each topic
Tau = zeros(F,T);
for t=1:T
    Tau(:,t) = drchrnd(Beta',1)';
end

% Initialize the group distribution.
%Pi = drchrnd(Eta',1)';
Pi = ones(1,K)/K;

% Initialize Y's to random
MNRnd = mnrnd(1,Pi,N);
Y = arrayfun(@(x) find(MNRnd(x,:)==1), 1:size(MNRnd,1));
clear MNRnd

% Initialize Z's to random
Z = zeros(size(fvs,1),1);
j = 1;
for i=1:N
    MNRnd = mnrnd(1,Theta(:,Y(i)),Ni(i));
    Z(j:j+Ni(i)-1) = arrayfun(@(x) find(MNRnd(x,:)==1), 1:size(MNRnd,1));
    j = j+Ni(i);
end
clear MNRnd

% Below we prepare a series of precomputed data structures.
% By convention, we will name all action counts prefixed with N*,
% all activity counts prefixed by M*, and all user counts prefixed
% with U*.

% Precompute number of users in each group
Uk = histc(Y,1:K);

% Precompute number of activities in each topic for each user
Mti = zeros(T,N);
j = 1;
for i=1:N
    Mti(:,i) = histc(Z(j:(j+Ni(i)-1)),1:T);
    j = j+Ni(i);
end

% Precompute number of actions in each activity.
Nw = sum(fvs(:,2:end),2);

% Precompute the count of each activity and each action in each group
Mkt = zeros(K,T); % No. of activities per topic per group
Nkt = zeros(K,T); % No. of actions per topic per group
Nvtk = zeros(F,T,K); % Counts per action per topic per group

% Following are at user-level
Nit = zeros(N,T); % Total no. of actions per topic per user
Nvti = zeros(F,T,N); % Counts for each action per topic per user

j = 1;
for i=1:N
    k = Y(i);
    fvs_i = fvs(fvs(:,1)==i,2:end);
    Zi = Z(j:(j+Ni(i)-1));
    for t=1:T
        IX = find(Zi==t);
        if ~isempty(IX)
            Mkt(k,t) = Mkt(k,t) + size(IX,1);
            Tmp = sum(fvs_i(IX,:),1);
            tot = sum(Tmp);
            Nkt(k,t) = Nkt(k,t) + tot;
            Nit(i,t) = Nit(i,t) + tot;
            Nvtk(:,t,k) = Nvtk(:,t,k) + Tmp';
            Nvti(:,t,i) = Nvti(:,t,i) + Tmp';
        end
    end
    j = j+Ni(i);
end
clear Tmp fvs_i

TauHist = [];
ThetaHist = [];
PiHist = [];

% Initialize all indexes having non-zero values
len = size(fvs,1);
M = cell(len,1);
SV = cell(len,1);
for i=1:len
    fv = fvs(i,2:end);
    M{i,1} = find(fv>0);     % Sparse vector indexes
    SV{i,1} = fv(:,M{i,1})'; % Sparse vector values
end

%%

% Start the Collapsed Gibbs Sampling step

rng('default');
%rand('seed',43); % Octave

for epoch=1:100

    p = 1;
    for i=1:N
        
        for j=1:Ni(i) % Each document of i-th user
            
            %-----------------------------------------------
            % Gibbs Sampling for group
            %-----------------------------------------------
            k = Y(i); % group of current user
            Mkt(k,:) = Mkt(k,:) - Mti(:,i)';
            Uk(k) = Uk(k) - 1;

            % Compute p(yi = k | Y_(-i), Zi, Z_(-i), W)
            Pk = zeros(1,K);
            Ar1 = 0:(sum(Mti(:,i))-1);
            pn = zeros(size(Ar1));
            for kk=1:K
                tpos = 1;
                pd = Ar1 + sum( Mkt(kk,:) + Alpha' );
                for t=1:T
                    if Mti(t,i) > 0
                        pn(1,tpos:(tpos+Mti(t,i)-1)) = ...
                            ( 0:(Mti(t,i)-1) ) + ( Mkt(kk,t) + Alpha(t) );
                    end
                    tpos = tpos + Mti(t,i);
                end
                Pk(kk) = sum(log(pn ./ pd)) + log(Uk(kk)+Eta(kk));
            end
            Pk = Pk - min(Pk);
            Pk = exp(Pk) / sum(exp(Pk));

            % Sample new group k
            nk = find(mnrnd(1,Pk,1)==1);

            % Adjust the values to new k
            Mkt(nk,:) = Mkt(nk,:) + Mti(:,i)';
            Uk(nk) = Uk(nk) + 1;
            Y(i) = nk;

            % In case a group changes, other pre-computed values need
            % adjustment.
            %if (k ~= nk)
                % subtract from old group
                Nkt(k,:) = Nkt(k,:) - Nit(i,:);
                Nvtk(:,:,k) = Nvtk(:,:,k) - Nvti(:,:,i);

                % add to new group
                Nkt(nk,:) = Nkt(nk,:) + Nit(i,:);
                Nvtk(:,:,nk) = Nvtk(:,:,nk) + Nvti(:,:,i);
            %end

            if (min(min(Nvtk(:,:,k))) < 0)
                error ('Nvtk(:,:,k) is < 0 at sampling k');
            end

            k = nk;

            %-----------------------------------------------
            % Gibbs Sampling for topic
            %-----------------------------------------------
            % Compute p(zi = t | Z_(-i), W)
            
            idx = p+j-1;
            %fv = fvs(idx,2:end); % curr doc -- j-th document of i-th user
            z = Z(idx);           % label of document
            l = Nw(idx);          % Number of words in curr document
            m = M{idx,1};
            sv = SV{idx,1};
            
            % Adjusted word counts for z-th topic excluding curr doc
            Nvtk(m,z,k) = Nvtk(m,z,k) - sv;
            Nvti(m,z,i) = Nvti(m,z,i) - sv;
            Nkt(k,z) = Nkt(k,z) - l;
            Nit(i,z) = Nit(i,z) - l;

            % Adjusted document counts for z-th topic excluding curr doc
            Mkt(k,z) = Mkt(k,z) - 1;
            Mti(z,i) = Mti(z,i) - 1;
            
            if (min(min(Nvtk(:,:,k))) < 0)
                error ('Nvtk(:,:,k) is < 0 at sampling t')
            end
            
            % Numerator of p(zi = k | Z_(-i), W)
            P = zeros(1,T);
            for t=1:T
                P(t) = 1;
                ll = 0;
                for mm=1:size(m,2)
                    for ssv=1:sv(mm);
                        P(t) = P(t)*(Nvtk(m(mm),t,k)+Beta(m(mm))+ssv-1)/(Nkt(k,t)+BetaSum+ll);
                        ll = ll+1;
                    end
                end
            end
            P = P / sum(P);

            % sample new z
            nz = find(mnrnd(1,P,1)==1);

            % Readjusted document counts for k-th topic including curr doc
            Mti(nz,i) = Mti(nz,i) + 1;
            Mkt(k,nz) = Mkt(k,nz) + 1;

            % Readjusted word counts for k-th topic including curr doc
            Nit(i,nz) = Nit(i,nz) + l;
            Nkt(k,nz) = Nkt(k,nz) + l;
            Nvti(m,nz,i) = Nvti(m,nz,i) + sv;
            Nvtk(m,nz,k) = Nvtk(m,nz,k) + sv;

            % Change the label for curr doc
            Z(idx) = nz;

        end
        fprintf('Completed Gibbs Sampling Epoch %d User %d...\n',epoch,i);
        %fprintf('Completed Gibbs Sampling Epoch %d User %d...[%d, %d, %d, %d]\n',...
        %    epoch,i,sum(sum(Nkt)),sum(sum(Nit)),sum(sum(sum(Nvtk))),sum(sum(sum(Nvti))));
        p = p+Ni(i);
    end

    for t=1:T
        Tau(:,t) = 0;
        for k=1:K
            Tau(:,t) = Tau(:,t) + Nvtk(:,t,k);
        end
        Tau(:,t) = (Tau(:,t)+Beta) ./ sum(Tau(:,t)+Beta);
    end
    for k=1:K
        Theta(:,k) = (Mkt(k,:)'+Alpha) ./ sum(Mkt(k,:)'+Alpha);
    end
    Pi = histc(Y,1:K);
    Pi = Pi/sum(Pi);
    
    TauHist = [TauHist; [ones(F,1)*epoch Tau]];
    ThetaHist = [ThetaHist; [ones(T,1)*epoch Theta]];
    PiHist = [PiHist; [epoch Pi]];
    %disp(Theta);
    
    fprintf('Completed Gibbs Sampling Epoch %d...\n',epoch);

end

%%

% Check whether all the counts are consistent

sum(sum(fvs(:,2:end))) % Actual total action count
size(fvs,1) % Actual total activity count
sum(Nw)

sum(sum(Nkt))
sum(sum(Nit))
sum(sum(sum(Nvtk)))
sum(sum(sum(Nvti)))

sum(sum(Mkt))
sum(sum(Mti))

%%

save(strcat(outpath,'/inferred_params_anom.mat'), 'TauHist', 'ThetaHist', ...
    'PiHist', 'Tau', 'Theta', 'Pi', 'Z', 'Y');

% Save to csv
csvwrite(strcat(outpath,'/anom_user_5_ThetaHist.csv'),ThetaHist);
csvwrite(strcat(outpath,'/anom_user_5_TauHist.csv'),TauHist);
csvwrite(strcat(outpath,'/anom_user_5_PiHist.csv'),PiHist);
csvwrite(strcat(outpath,'/anom_user_5_Z.csv'),Z);
csvwrite(strcat(outpath,'/anom_user_5_Y.csv'),Y');

clear TauHist ThetaHist PiHist

%%

load(strcat(outpath,'/inferred_params_anom.mat'));

%%

% Analyze Convergence of Theta
A = ThetaHist(:,2); % Convergence of one parameter
s = size(A,1)/3-2;
plot(0:s,A((0:s)*3+1));

% Analyze Convergence of Pi
s = size(PiHist,1);
plot(1:s,PiHist(:,3));

Theta

plot(1:F,Tau(:,1),'r','Linewidth',2)
hold on
plot(1:F,Tau(:,2),'b','Linewidth',2)
hold on
plot(1:F,Tau(:,3),'k','Linewidth',2)
hold on
