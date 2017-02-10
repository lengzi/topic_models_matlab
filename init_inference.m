%%
% Setup environment
clear all

outpath = 'data/MATLAB-data';
inpath  = 'data/MATLAB-data';

%outpath = 'data/MATLAB-data';
%inpath  = 'data/MATLAB-data';
%%

load(strcat(inpath,'/fv_data_9000.mat'));

%load(strcat(inpath,'/fv_data_anom_user_5.csv'));
%fvs = fv_data_anom_user_5;
%clear fv_data_anom_user_5

%%
% Check if any column other than the first has values > 1.
% This makes sure all features are boolean.
for i=1:size(fvs,2)
    tmp = find(fvs(:,i) > 1);
    if isempty(tmp)
        %disp(strcat('Empty: ',num2str(i)));
    else
        disp(strcat('Not Empty: ',num2str(i)));
    end
end

