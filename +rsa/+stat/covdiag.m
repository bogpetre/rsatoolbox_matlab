function [sigma,shrinkage,sample]=covdiag(x,df,varargin)
% function [sigma,shrinkage]=covdiag(x,df)
% Regularises the estimate of the covariance matrix accroding to the 
% optimal shrikage method as outline in Ledoit& Wolf (2005). 
% Shrinks towards diagonal matrix
% INPUT: 
%       x     (t*n): t observations on p random variables
%       df    (1*1): optional: degrees of freedom, if not t-1 
% OUTPUT: 
%       sigma (n*n): invertible covariance matrix estimator
%       shrinkage  : Applied shrinkage factor 
% 

Opt.shrinkage = []; 
Opt.target = [];
Opt = rsa.getUserOptions(varargin,Opt);

% de-mean returns
[t,n]=size(x);
x=bsxfun(@minus,x,sum(x)./t);  % Subtract mean of each time series fast  

% compute sample covariance matrix
if nargin<2 || isempty(df) 
    df=t-1; 
end; 

% Compute sample covariance matrix
sample=(1/df).*(x'*x);

% compute prior
if isempty(Opt.target) || strcmp(Opt.target,'diagonal')
    prior=diag(diag(sample));
elseif strcmp(Opt.target,'scaledidentity')
    prior = mean(diag(sample))*eye(size(sample,2));
else
    error('Did not understand target type %s', Opt.target);
end

% if required, compute shrinkage parameter using Ledoit-Wolf method 
if (isempty(Opt.shrinkage))
    %{
    d=1/n*norm(sample-prior,'fro')^2;
    y=x.^2;
    r2=1/n/df^2*sum(sum(y'*y))-1/n/df*sum(sum(sample.^2));
    shrinkage = max(0,min(1,r2/d));
    %}
    % simplify for legibility:
    d=norm(sample-prior,'fro')^2;
    y=x.^2;
    % r2 this is "b2" in the parlance of Leidot and Wolf 2004 J of Multivar
    % Analysis, but note that they estimate this with a scaled identity
    % matrix as their prior, not a diagonal matrix, and it's not obvious 
    % that this generalizes.
    r2=1/df^2*sum(sum(y'*y))-1/df*sum(sum(sample.^2));
    
    shrinkage = max(0,min(1,r2/d));

    % the above only works well if various assumptions are met including iid
    % assumptions, which are hard to meet with timeseries fMRI data. As a fall
    % back let's add another constraint. This constraint falls to zero iff we
    % have 3x as many df as parameters, a consrevative threshold. At that point
    % we let the ledoit wolf estimator take over completely. This assumes our
    % prior has n parameters of its own, and that if we have less than 3x as
    % many df as we have dimensions we can't even estimate that prior well.
    %{
    lowerbound = min(1, 3*n/(df));
    
    shrinkage = max(lowerbound, shrinkage);
    %}
else 
    shrinkage = Opt.shrinkage; 
end; 

% Regularize the estimate
sigma=shrinkage*prior+(1-shrinkage)*sample;

