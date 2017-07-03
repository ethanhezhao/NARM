
function model = AsymNARM(B, F, K, xitrain, idtrain, xitest,idtest, Burnin, Collections)

%Code for AsymNARM, with some functions from Mingyuan Zhou's EPM code and Changwei Hu's SSI-PF code.
%He Zhao, FIT, Monash University
%June, 2017

%Input:
%B is a matrix, the diagonal terms are not defined
%F is a matrix, each row is a node's binary attribute vector
%idx_train, xitrain: training indices and values
%idx_test, xitest: test indices and values
%K: truncated number of atoms

%Output:
%model.Phi: the node factor loading matrix
%model.g: the prior of the node factor loading matrix
%model.h: the attribute factor loading matrix
%model.t: the tables for augmenting the gamma ratio
%model.Theta: the loading score matrix
%model.q: q(k) is the weight of latent factor $k$
%model.x_i_k: x_i_k(i,k) is the count that node $i$ sends to community (latent factor) $k$ when node $i$ is a head node
%model.n_j_k: n_j_k(j,k) is the count that node $j$ sends to community (latent factor) $k$ when node $j$ is a tail node
%model.AUCroc: area under the ROC curve
%model.AUCpr: area under the precition-recall curve
%model.ProbAve: ProbAve(i,j) is the estimated probability for nodes $i$ and $j$ to be linked

if ~exist('K','var')
    K = 100;
end
if ~exist('Burnin','var')
    Burnin = 1000;
end
if ~exist('Collections','var')
    Collections = 500;
end
iterMax = Burnin+Collections;


[N,~]=size(B);

F = [F,ones(N,1)];

L = size(F,2);


alpha_0 = 0.1;



h = 1.0 * ones(L, K);

g = exp(F * log(h));

mu_0 = 1;

nu_0 = 1 ./ mu_0;

active_nodes = cell(L);

for l = 1:L
    active_nodes{l} = find(F(:,l));
end



Theta = randg(alpha_0 .* ones(N,K));

Theta = bsxfun(@times, Theta, 1./(sum(Theta, 1)));


c=1;
epsi=1/K;

q=betarnd(c*epsi*ones(1,K),c*(1-epsi)*ones(1,K));

Phi = randg(g) .* q ./ (1 - q);




ProbSamples = 0;


for iter=1:iterMax
    
    x_i_k = zeros(N,K);
    
    
    n_j_k = zeros(N,K);
        
    for ii=1:length(xitrain)

        j = idtrain(ii,2);

        i = idtrain(ii,1);

        Rate=Theta(j,:).*Phi(i,:)+eps;

        sum_cum = cumsum(Rate(:));



        M=truncated_Poisson_rnd_1(1,sum_cum(end));


        for r = 1:M
            k = find(sum_cum > rand() * sum_cum(end),1);
            n_j_k(j,k) = n_j_k(j,k) + 1;
            x_i_k(i,k) = x_i_k(i,k) + 1;
        end

    end

    
    

    Theta = randg(alpha_0 + n_j_k);

    Theta = bsxfun(@times, Theta, 1./(sum(Theta, 1)));
    
    
    
    
    t = zeros(N,K);
    
    t(x_i_k>0) = 1;
    
    for i = 1:N        
        for k = 1:K
            for j=1:x_i_k(i,k)-1
                t(i,k) = t(i,k) + double(rand() < g(i,k) ./ (g(i,k) + j));
            end
        end
    end
    
    log_q = - log( 1 - q);
    
    
    new_h = randg(mu_0 + F' * t);
    
    
    
    for l = 1:L
        
        an_l = active_nodes{l};
        
        new_h_l = new_h(l,:) ./ (sum(g(an_l, :) .* log_q, 1) + nu_0 * h(l,:));
        
        g(an_l, :) = g(an_l,:) .* new_h_l;
        
        h(l,:) = new_h_l .* h(l,:);
        
    end
    
    Phi = randg(g + x_i_k) .* q + eps;  
        
    
    q=betarnd(c*epsi+sum(x_i_k,1),c*(1-epsi)+ sum(g,1));
    
    
    
    Prob =sum(Theta(idtest(:,2),:).*Phi(idtest(:,1),:),2)+eps;
    
   
    Prob = 1-exp(-Prob);
    if iter > Burnin
        ProbSamples = ProbSamples +  Prob;
        ProbAve = ProbSamples ./ (iter - Burnin);
    else
        ProbAve = Prob;
    end
    
    
end

AUCroc = compute_AUC(xitest,ProbAve,ones(1,length(xitest)));
[prec, tpr, ~, ~] = prec_rec(ProbAve, xitest);

AUCpr = trapz([0;tpr],[1;prec]);

model.AUCroc = AUCroc;

model.AUCpr = AUCpr;

model.ProbAve = ProbAve;

model.h = h;

model.g = g;

model.Theta = Theta;

model.Phi = Phi;

model.q_k = q;

model.t = t;

model.x_i_k = x_i_k;

model.n_j_k = n_j_k;







