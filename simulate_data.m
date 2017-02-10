%% Initialize
clear all

outpath = 'data/MATLAB-data';
inpath  = 'data/MATLAB-data';
%% Setup Action Data

rng('default');

% Assume:
%  - 5 actions {copy, delete, modify, email, terminate}
%  - 5 activity-sequences with action-probabilities {T1,...,T5}
% The transition matrices show 

% State names
snames = {'S','C','D','M','E','T'};

% Editing
T1 = [
    0.00, 0.14, 0.05, 0.76, 0.05, 0.00; % start
    0.00, 0.05, 0.05, 0.30, 0.05, 0.55; % copy
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % delete
    0.00, 0.05, 0.05, 0.50, 0.05, 0.35; % modify
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % email
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % terminate
    ];

% Backing Up
T2 = [
    0.00, 0.94, 0.03, 0.02, 0.01, 0.00; % start
    0.00, 0.05, 0.05, 0.01, 0.05, 0.84; % copy
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % delete
    0.00, 0.42, 0.01, 0.01, 0.01, 0.55; % modify
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % email
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % terminate
    ];

% Copy-Email-Delete
T3 = [
    0.00, 0.94, 0.03, 0.02, 0.01, 0.00; % start
    0.00, 0.05, 0.05, 0.01, 0.74, 0.15; % copy
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % delete
    0.00, 0.10, 0.11, 0.01, 0.53, 0.25; % modify
    0.00, 0.00, 0.80, 0.00, 0.00, 0.20; % email
    0.00, 0.00, 0.00, 0.00, 0.00, 1.00; % terminate
    ];

TM = {T1, T2, T3};

% Probabilities for users
P1 = [0.30, 0.55, 0.15];
P2 = [0.55, 0.30, 0.15];
P3 = [0.15, 0.20, 0.65];

%P = {P1,P2,P3};
P = [repmat(P1,30,1); repmat(P2,30,1); repmat(P3,2,1)];

nusers = size(P,1);
n = 300;
Ni = 1+[poissrnd(n,[1 30]) poissrnd(n,[1 30]) poissrnd(50,[1 2])];

%%

data = {};
for i=1:nusers
    data{i} = action_seq_user(Ni(i),P(i,:),TM,snames,strcat('U',num2str(i),'_'));
    fprintf('Generated sequences for user %d\n',i);
end;

%%

save(strcat(outpath,'/synthetic_data.mat'), 'data', 'n', 'nusers', 'P', 'TM', 'snames');

%%
% Enumerate all unique states

state2id = java.util.Hashtable; % lookup state by name
id2state = java.util.Hashtable; % lookup state by id
statecounts = java.util.Hashtable;
maxstates = 1;
for i=1:nusers
    nseqs = length(data{i});
    %fprintf('User %d sequences: %d\n',i,nseqs);
    for j=1:nseqs
        data_user_seq = data{i}{j};
        states = data_user_seq.states;
        for k=1:length(states)
            if (state2id.containsKey(states{k}))
                val = statecounts.get(states{k});
                statecounts.put(states{k},val+1);
            else
                state2id.put(states{k},maxstates);
                id2state.put(maxstates,states{k});
                statecounts.put(states{k},1);
                maxstates = maxstates+1;
            end;
        end;
    end;
    fprintf('Enumerated sequences for user %d\n',i);
end;

%%
% Translate the user sequences to feature vectors

cols = state2id.size();
fvs = [];
for i=1:nusers
    nseqs = length(data{i});
    user_fvs = zeros(nseqs, cols);
    for j=1:nseqs
        data_user_seq = data{i}{j};
        states = data_user_seq.states;
        for k=1:length(states)
            stateid = state2id.get(states{k});
            user_fvs(j,stateid) = user_fvs(j,stateid) + 1;
        end
    end
    fvs = [fvs; [ones(nseqs, 1)*i user_fvs]];
    fprintf('Translated to feature vectors for user %d\n',i);
end

%%

save (strcat(outpath,'/fv_data.mat'), 'fvs', 'state2id', 'id2state', 'statecounts');

%%

%a = [2.1 3.2 4.3];
%n = 1;
%r = drchrnd(a,n)

%a = 2*[0.30, 0.70];
%n = 10;
%r = drchrnd(a,n)
%clear a n r
