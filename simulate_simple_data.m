%%
clear all

outpath = 'data/MATLAB-data';
inpath  = 'data/MATLAB-data';
%%

rng('default');

% Probabilities for users
%P1 = [0.30, 0.55, 0.15];
%P2 = [0.55, 0.30, 0.15];
%P3 = [0.15, 0.20, 0.65];

% Probabilities for users
P1 = [0.15, 0.70, 0.15];
P2 = [0.70, 0.15, 0.15];
P3 = [0.15, 0.15, 0.70];

% Anomalous user
AnomalousUser = 5;
NAnomalous = 15; % No. of anomalous data points

% No. features
F = 50;

% Probabilities for Tau's
Beta = ones(F,1);
Beta = Beta ./ sum(Beta);
% To see higher precision: format longG, else format short
Tau = zeros(F,3);
Tau(:,1) = exp(-(1:F)*0.1);
Tau(:,2) = Tau((F:-1:1),1);
Tau(:,3) = [Tau((F/2):-1:1,1); Tau(1:(F/2),1)];
Tau(:,1) = Tau(:,1)/sum(Tau(:,1));
Tau(:,2) = Tau(:,2)/sum(Tau(:,2));
Tau(:,3) = Tau(:,3)/sum(Tau(:,3));

%AnomalousTau = [Tau(1:(F/2),1); Tau((F/2):-1:1,1)]; % A U-shaped dist

% The below generates a distribution with two spikes
AnomalousTau = [zeros(floor(F/4),1); exp(-(1:ceil(F/4))*0.45)'];
AnomalousTau = [AnomalousTau; AnomalousTau];
AnomalousTau = AnomalousTau + AnomalousTau(F:-1:1);
AnomalousTau = AnomalousTau/sum(AnomalousTau);

plot_tau = 1;
if plot_tau == 1
    % plot all topic feature distributions
    plot(1:F,Tau(:,1),'k','Linewidth',2);
    hold on;
    plot(1:F,Tau(:,2),'b','Linewidth',2);
    plot(1:F,Tau(:,3),'g','Linewidth',2);
    plot(1:F,AnomalousTau,'r','Linewidth',2);
    hold off;
end

%P = {P1,P2,P3};
P = [repmat(P1,10,1); repmat(P2,10,1); repmat(P3,10,1)];

nusers = size(P,1);
n = 300;

%%
% -- Simulate binary multinomial
rng('default');

fvs = zeros(nusers*n,F+1);
l = 1;
for i=1:nusers
    p = P(i,:);
    for j=1:n
        z = find(mnrnd(1,p,1)==1);
        w = find(mnrnd(1,Tau(:,z),1)==1);
        fvs(l+j-1,w+1) = 1;
        fvs(l+j-1,  1) = i;
    end
    l = l+n;
end

%%
% -- Simulate true multinomial
rng('default');
meanDocLen = 6;
meanAnomDocLen = 9;
fvs = zeros(nusers*n,F+1);
l = 1;
for i=1:nusers
    p = P(i,:);
    for j=1:n
        z = find(mnrnd(1,p,1)==1);
        numw = poissrnd(meanDocLen,1,1)+1;
        fvs(l+j-1,2:end) = mnrnd(numw,Tau(:,z)',1);
        fvs(l+j-1,  1) = i;
    end
    l = l+n;
    if i==AnomalousUser
        for j=1:NAnomalous
            numw = poissrnd(meanAnomDocLen,1,1)+1;
            fvs(l+j-1,2:end) = mnrnd(numw,AnomalousTau',1);
            fvs(l+j-1,  1) = i;
        end
        l = l+NAnomalous;
    end
end

%%

save(strcat(outpath,'/fv_data_anom_user_5.mat'),'Tau','P','Beta','F','nusers','n','fvs');
csvwrite(strcat(outpath,'/fv_data_anom_user_5.csv'),fvs);
