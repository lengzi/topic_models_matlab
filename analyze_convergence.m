%%
clear all

outpath = 'results/lda';
inpath  = 'data/MATLAB-data';

%%

fvs = csvread(strcat(inpath,'/fv_data_anom_user_5.csv'));
ThetaHist = csvread(strcat(outpath,'/fv_data_anom_user_5_thetas_T3.csv'));
TauHist   = csvread(strcat(outpath,'/fv_data_anom_user_5_taus_T3.csv'));

%%

last_epoch = max(ThetaHist(:,1));

Theta = ThetaHist(ThetaHist(:,1)==last_epoch,2:end);
Tau   = TauHist(TauHist(:,1)==last_epoch,2:end);
Tau   = TauHist(TauHist(:,1)==last_epoch,2:end);

F = size(Tau,1);
T = size(Tau,2);

%% Fit a Gaussian Distribution to the ThetaHist

obj = gmdistribution.fit(Theta_last',3);

%%

% plot the values over iterations to check if MCMC converged
plot_convergence = 0;
if plot_convergence == 1
    idxs = (0:last_epoch)*F;
    for i=1:F
        Tmp = TauHist(idxs+i,2);
        subplot(5,10,i);
        plot(0:last_epoch,Tmp);
    end
end

avgn = 10;
start_epoch = last_epoch-avgn+1; % take average of last 10 epochs only
ThetasX = ThetaHist(ThetaHist(:,1)>=start_epoch,2:end);
TausX   = TauHist(TauHist(:,1)>=start_epoch,2:end);

Tau = zeros(F,T);
idxs = (0:(avgn-1))*F;
for i=1:F
    Tmp = TausX(idxs+i,:);
    Tau(i,:) = mean(Tmp);
end

Theta = zeros(size(Theta_last));
idxs = (0:(avgn-1))*size(Theta,1);
for i=1:size(Theta,1)
    Tmp = ThetasX(idxs+i,:);
    Theta(i,:) = mean(Tmp);
end

% Plot the topic distributions
plot_tau = 0; % set to 1 for plot
if plot_tau == 1
    % plot all topic feature distributions
    plot(1:F,Tau(:,1),'k','Linewidth',2);
    hold on;
    for i=2:T
        plot(1:F,Tau(:,i),'k','Linewidth',2);
        %plot(1:F,Tau(:,3),'g','Linewidth',2);
    end
    %plot(1:F,AnomalousTau,'r','Linewidth',2);
    hold off;
end

%% Calculate the probabilities of each document

N = size(fvs,1);
P = zeros(N,1);
%Pk = zeros(N,K);
Pt = zeros(N,T);
T = size(Theta,1);
K = size(Theta,2);
for i=1:N
    uid = fvs(i,1);
    fv = fvs(i,2:end);
    m = find(fv>0);
    val = fv(m);
    k = Y(uid);
    %for k=1:K
    for t=1:T
        Pt(i,t) = sum(log(Tau(m,t))' .* val)/sum(val);
    end
    P(i) = sum(exp(Pt(i,:))'.*Theta(:,k));
    %end
    %P(i,1) = sum(Pi.*Pk(i,:));
end
[B, IX] = sort(P);
[fvs(IX(1:100),1) P(IX(1:100)) IX(1:100)]

A = find(IX>1500 & IX<1516);
IX(A)
AUC = zeros(size(IX,1),1);
AUC(A) = 1;
m = sum(AUC);
AUC = cumsum(AUC);
AUC = AUC / m;
plot(1:size(AUC,1),AUC);

%% Analyze the count of words in top ranked docs

lSz = length(find(Nw>8));
RankedAUCNw = Nw(IX);
ANw = RankedAUCNw(1:lSz);
alSz = length(find(ANw>8));


%% Analyze convergence for GMM
gmminpath = 'results/synthetic';
gmmdata = csvread(strcat(gmminpath,'/fv_data_anom_user_5_id_ranked_no_colname.csv'));
IX = gmmdata(:,2);
A = find(IX>1500 & IX<1516);
IX(A)
AUC = zeros(size(IX,1),1);
AUC(A) = 1;
m = sum(AUC);
AUC = cumsum(AUC);
AUC = AUC / m;
plot(1:size(AUC,1),AUC);
