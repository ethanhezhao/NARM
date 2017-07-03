

addpath(genpath('.'));

load './datasets/facebook-ego.mat';


TrainRatio = 0.1;

[idx_train,idx_test,BTrain_Mask] = Create_Mask_network(B, TrainRatio);

K = 100;

[AUCroc,AUCpr,F1,Phi,Lambda_KK,r_k,ProbAve,m_i_k_dot_dot,output,z]=HGP_EPM(B,K, idx_train,idx_test,1500, 1500,false);
fprintf('HGP_EPM, AUCroc =  %.4f, AUCpr = %.4f\n',AUCroc,AUCpr);

model = SymNARM(B, F, K, idx_train, idx_test);
fprintf('SymNARM, AUCroc =  %.4f, AUCpr = %.4f\n',model.AUCroc,model.AUCpr);




