% Global variables

T = 3;
K = 3;
U = sort(unique(fvs(:,1)));
N = length(U); % No. users
F = size(fvs,2)-1; % The first column is the userid

%% Standardize by mean and variance

fvs_n = fvs;
fvs_n(:,1) = 0; % Zero out user ids for easy mean centering
MuHat = mean(fvs_n);
for i=1:size(fvs_n)
    fvs_n(i,:) = fvs_n(i,:) - MuHat; % Mean Centered
end

for i=2:(F+1)
    sdf = sqrt(var(fvs_n(:,i)));
    fvs_n(:,i) = fvs_n(:,i)/sdf; % Features are Variance normalized
end

fvs_n(:,1) = fvs(:,1); % Put back user ids
fvs = fvs_n;

clear fvs_n MuHat sdf

%%

% We will assume our topic model has T topics, N users, F features, K
% groups, and Ni action-sequences for the i-th user

% IMPORTANT: We assume that the input feature vectors are sorted by userids

%pkg load 'statistics' % Octave

%rng('default'); % MATLAB
%rand('seed',43); % Octave

Ni = zeros(size(U,1),1);
for i=1:N
    Ni(i) = length(find(fvs(:,1)==U(i)));
end

% Initialize the group distribution.
%Pi = drchrnd(Eta',1)';
Pi = ones(1,K)/K;

% Initialize Y's to random
Y = mnrnd(1,Pi,N);

%% E-M Algorithm

for epoch=1:5

    %--------
    % M-Step
    %--------

    % Compute Mu_k
    Mu = zeros(K,F);
    for k=1:K
        p = 1;
        denom = 0;
        numer = zeros(1,F);
        for i=1:N
            fvs_i = fvs(p:(p+Ni(i)-1),2:(F+1));
            numer = numer + Y(i,k)*sum(fvs_i);
            denom = denom + Ni(i)*Y(i,k);
            p = p+Ni(i);
        end
        Mu(k,:) = numer / denom;
    end
    clear fvs_i numer denom

    % Compute Sigma_k
    Sigma = zeros(F,F,K);
    for k=1:K
        p = 1;
        denom = zeros(F,F);
        numer = zeros(F,F);
        for i=1:N
            fvs_i = fvs(p:(p+Ni(i)-1),2:(F+1));
            MuK = repmat(Mu(k,:),Ni(i),1);
            fvs_i = fvs_i - MuK;
            numer_i = zeros(F,F);
            for j=1:Ni(i)
                numer_i = numer_i + (fvs_i(j,:)'*fvs_i(j,:));
            end
            numer = numer + Y(i,k) * numer_i;
            denom = denom + Ni(i)*Y(i,k)*eye(F);
            p = p+Ni(i);
        end
        Sigma(:,:,k) = numer / denom;
    end
    clear fvs_i MuK denom numer numer_i

    % Compute Pi
    denom = sum(sum(Y));
    for k=1:K
        Pi(k) = sum(Y(:,k))/denom;
        Pi(k) = max(Pi(k),1e-300);
    end
    Pi = Pi ./ sum(Pi);
    clear denom

    %--------
    % E-Step
    %--------

    % Compute Y_ik
    iSigma = zeros(F,F,K);
    LogSqrtDet = zeros(1,K);
    SvdU = zeros(F,F,K);
    for k=1:K
        % since direct inversion is not safe because of singular
        % matrices, we will user SVD
        [u, s, ~] = svd(Sigma(:,:,k)); % ignore the 'v' returned by SVD
        s = diag(s); % get the diagonal elements
        s = max(s,1e-300); % Handle zero eigen values
        SvdU(:,:,k) = u;
        LogSqrtDet(k) = log(sqrt(prod(s)));
        iSigma(:,:,k) = inv(diag(s));
    end
    clear u s v
    p = 1;
    for i=1:N
        Pk = zeros(1,K);
        fvs_i = fvs(p:(p+Ni(i)-1),2:(F+1));
        for k=1:K
            MuK = repmat(Mu(k,:),Ni(i),1);
            fvs_ik = fvs_i - MuK;
            iSigmaK = iSigma(:,:,k);
            Pk(k) = log(Pi(k)) - LogSqrtDet(k)*Ni(i);
            for j=1:Ni(i)
                % x'*Sigma*x = x'*u*SigSVD*u'*x = (x'u)*SigSVD*(x'u)'
                centered = fvs_ik(j,:) * SvdU(:,:,k);
                Pk(k) = Pk(k) - 0.5*(centered * iSigmaK * centered');
            end
        end
        %------ Handle Numerical Errors (Too large and too small)
        Pk = Pk - mean(Pk);
        if (max(Pk) > 700)
            diff = max(Pk) - 700; % Too large cannot be ignored...
            Pk = Pk - diff;
        end
        %Pk
        %-------
        Pk = exp(Pk);
        Pk = Pk ./ sum(Pk);
        Y(i,:) = Pk;
        p = p+Ni(i);
    end
    clear fvs_i fvs_ik MuK denom iSigmaK Pk centered

    fprintf('Completed E-M Epoch %d...\n',epoch);
end
