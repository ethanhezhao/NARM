function model = SymNARM(B, F, K, idx_train, idx_test, Burnin, Collections)

%Code for SymNARM, built on top of Mingyuan Zhou's EPM code.
%He Zhao, FIT, Monash University
%June, 2017

%Input:
%B is an upper triagular matrix, the diagonal terms are not defined
%F is a matrix, each row is a node's binary attribute vector  
%idx_train: training indices
%idx_test: test indices
%K: truncated number of atoms

%Output:
%model.Phi: the node factor loading matrix
%model.g: the prior of the node factor loading matrix
%model.h: the attribute factor loading matrix
%model.t: the tables for augmenting the gamma ratio
%EPM
%model.m_i_k_dot_dot: m_i_k_dot_dot(k,i) is the count that node $i$ sends to community (latent factor) $k$
%model.r_k: each elment indicates a community's popularity
%model.Lambda_KK: community-community interaction strengh rate matrix
%model.AUCroc: area under the ROC curve
%model.AUCpr: area under the precition-recall curve
%model.ProbAve: ProbAve(i,j) is the estimated probability for nodes $i$ and $j$ to be linked
% rng(1)

if ~exist('K','var')
    K = 100;
end
if ~exist('Burnin','var')
    Burnin = 1500;
end
if ~exist('Collections','var')
    Collections = 1500;
end


iterMax = Burnin+Collections;

N = size(B,2);

BTrain_Mask = zeros(size(B));
BTrain_Mask(idx_train) = 1;
BTrain_Mask=BTrain_Mask+BTrain_Mask';

BTrain = B;
BTrain(idx_test)= 0;

[ii,jj,~]=find(BTrain);

ProbSamples = zeros(N,N);
count=0;
EPS=0;
IsMexOK=true;

links = double(B(idx_test)>0);

%% Parameter initialization
Epsilon = 1;
beta1=1;
beta2=1;
c0=1;
e_0=1e-0;
f_0=1e-0;
gamma0=1;


Phi = gamrnd(1e-0*ones(N,K),1);
r_k=ones(K,1)/K;
Lambda_KK=r_k*r_k';
Lambda_KK = triu(Lambda_KK,1)+triu(Lambda_KK,1)';
Lambda_KK(sparse(1:K,1:K,true))=Epsilon*r_k;
c_i = ones(N,1);

%% Variables related to node attributes

F = [F,ones(N,1)]; %Add the default feature

L = size(F,2);

h = 1.0 * ones(L, K);

g = exp(F * log(h));

mu_0 = 1;

nu_0 = 1 ./ mu_0;

active_nodes = cell(L);

for l = 1:L
    active_nodes{l} = find(F(:,l));
end

%% Inference
for iter=1:iterMax
    
    %% EPM: draw a latent count for each edge
    Rate = sum((Phi(ii,:)*Lambda_KK).*Phi(jj,:),2);
    M = truncated_Poisson_rnd(Rate);

    %Sample m_i_k1_k2_j and update m_i_k_dot_dot and m_dot_k_k_dot
    if IsMexOK
        [m_i_k_dot_dot, m_dot_k_k_dot] = Multrnd_mik1k2j(sparse(ii,jj,M,N,N),Phi,Lambda_KK);
    else
        m_i_k_dot_dot = zeros(K,N);
        m_dot_k_k_dot = zeros(K,K);
        for ij=1:length(idx)
            pmf = (Phi(ii(ij),:)'*Phi(jj(ij),:)).*Lambda_KK;
            mij_kk = reshape(multrnd_histc(M(ij),pmf(:)),K,K);
            m_i_k_dot_dot(:,ii(ij)) = m_i_k_dot_dot(:,ii(ij)) + sum(mij_kk,2);
            m_i_k_dot_dot(:,jj(ij)) = m_i_k_dot_dot(:,jj(ij)) + sum(mij_kk,1)';
            m_dot_k_k_dot = m_dot_k_k_dot + mij_kk + mij_kk';
        end
    end
    m_dot_k_k_dot(sparse(1:K,1:K,true))=m_dot_k_k_dot(sparse(1:K,1:K,true))/2;
        
    Phi_times_Lambda_KK = Phi*Lambda_KK;

    
    
    %% Node attributes
    p = BTrain_Mask*Phi_times_Lambda_KK;

    %Sample tables by CRP
    t = zeros(N,K);
    t(m_i_k_dot_dot'>0) = 1;
    for i = 1:N
        for k = 1:K
            for j=1:m_i_k_dot_dot(k,i)-1
                t(i,k) = t(i,k) + double(rand() < g(i,k) ./ (g(i,k) + j));
            end
        end
    end
    
    log_p = log((p + c_i) ./(c_i));
    %Sample h and update g
    new_h = randg(mu_0 +  F' * t);
    for l = 1:L
        an_l = active_nodes{l};
        new_h_l = new_h(l,:) ./ (sum(g(an_l, :) .* log_p(an_l,:),1) + nu_0 * h(l,:));

        g(an_l, :) = g(an_l,:) .* new_h_l;

        h(l,:) = new_h_l .* h(l,:);

    end
    
    

    %% EPM: Sample phi_ik
    Phi_temp = randg(g + m_i_k_dot_dot');
    
    for i=randperm(N)
        Phi(i,:) =  Phi_temp(i,:)./(c_i(i)+BTrain_Mask(i,:)*Phi_times_Lambda_KK);
        Phi_times_Lambda_KK(i,:) = Phi(i,:)*Lambda_KK;
    end
        
    %% EPM: Sample c_i
    c_i = gamrnd(1e-0 + sum(g,2),1./(1e-0 +  sum(Phi,2)));

    Phi_KK = Phi'*BTrain_Mask*Phi;
    
    Phi_KK(sparse(1:K,1:K,true)) = Phi_KK(sparse(1:K,1:K,true))/2;
    
    triu1dex = triu(true(K),1);
    diagdex = sparse(1:K,1:K,true);
    
    
    %% EPM: Sample r_k
    L_KK=zeros(K,K);
    temp_p_tilde_k=zeros(K,1);
    p_kk_prime_one_minus = zeros(K,K);
    for k=randperm(K)
        R_KK=r_k';
        R_KK(k)=Epsilon;
        beta3=beta2*ones(1,K);
        beta3(k)=beta1;
        p_kk_prime_one_minus(k,:) = beta3./(beta3+ Phi_KK(k,:));
        
        L_KK(k,:) = CRT_sum_mex_matrix(sparse(m_dot_k_k_dot(k,:)),r_k(k)*R_KK);
        temp_p_tilde_k(k) = -sum(R_KK.*log(max(p_kk_prime_one_minus(k,:), realmin)));
        r_k(k) = randg(gamma0/K+sum(L_KK(k,:)))./(c0+temp_p_tilde_k(k));
    end
    
    %% EPM: Sample gamma0 with independence chain M-H
    ell_tilde = CRT_sum_mex(sum(L_KK,2),gamma0/K);
    sum_p_tilde_k_one_minus = -sum(log(c0./(c0+temp_p_tilde_k) ));
    gamma0new = randg(e_0 + ell_tilde)./(f_0 + 1/K*sum_p_tilde_k_one_minus);
    AcceptProb1 = CalAcceptProb1(r_k,c0,gamma0,gamma0new,ell_tilde,1/K*sum_p_tilde_k_one_minus,K);
    if AcceptProb1>rand(1)
        gamma0=gamma0new;
        count =count+1;
    end
    c0 = randg(1 + gamma0)/(1+sum(r_k));
    %% EPM: Sample Epsilon
    ell = sum(CRT_sum_mex_matrix( sparse(m_dot_k_k_dot(diagdex))',Epsilon*r_k'));
    Epsilon = randg(ell+1e-2)/(1e-2-sum(r_k.*log(max(p_kk_prime_one_minus(diagdex), realmin))));   
    
    %% EPM: Sample lambda_{k_1 k_2}
    R_KK = r_k*(r_k');
    R_KK(sparse(1:K,1:K,true)) = Epsilon*r_k;
    Lambda_KK=zeros(K,K);
    Lambda_KK(diagdex) = randg(m_dot_k_k_dot(diagdex) + R_KK(diagdex))./(beta1+Phi_KK(diagdex));
    Lambda_KK(triu1dex) = randg(m_dot_k_k_dot(triu1dex) + R_KK(triu1dex))./(beta2+Phi_KK(triu1dex));
    Lambda_KK = Lambda_KK + triu(Lambda_KK,1)'; %Lambda_KK is symmetric
    
    beta1 = randg(sum(R_KK(diagdex))+sum(R_KK(triu1dex))+1e-0)./(1e-0+ sum(Lambda_KK(diagdex))+sum(Lambda_KK(triu1dex)));
    beta2 = beta1;
    
    %% Compute and collect probabilites
    Prob =Phi*(Lambda_KK)*Phi'+EPS;    
    Prob = 1-exp(-Prob);
    if iter>Burnin
        
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples/(iter-Burnin);
    else
        ProbAve = Prob;
    end
    
%     if mod(iter,10) == 0
%         rate= ProbAve(idx_test);
%         disp(aucROC(rate,links));
%     end

end

%% Compute AUC-ROC and AUC-PR

rate = ProbAve(idx_test);

[~,~,~,AUCroc] = perfcurve(links,rate,1);

[prec, tpr, ~, ~] = prec_rec(rate, links,  'numThresh',3000);

AUCpr = trapz([0;tpr],[1;prec]);

%% Collect results

model.m_i_k_dot_dot = m_i_k_dot_dot;

model.Phi = Phi;

model.g = g;

model.h = h;

model.t = t;

model.c_i = c_i;

model.Lambda_KK = Lambda_KK;

model.r_k = r_k;

model.Epsilon = Epsilon;

model.AUCroc = AUCroc;

model.AUCpr = AUCpr;

model.ProbAve = ProbAve;


end






