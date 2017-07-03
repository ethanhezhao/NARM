

load './datasets/aminer.mat';


TrainRatio = 0.9;


[rows,cols,xi]=find(B);
id=[cols,rows];

[xitrain, idtrain, xitest,idtest]=datasplit(xi,id,TrainRatio);
[zeroID1,zeroID2] = ind2sub(size(B),find(B== 0));
zeroRatio=1;
zeroN=floor(length(xitest)*zeroRatio);
zeroIDid=randperm(length(zeroID1));
zeroIDid=zeroIDid(1:zeroN);
idtestadd=[zeroID2(zeroIDid),zeroID1(zeroIDid)];
idtest=[idtest;idtestadd];
xitest=[xitest;zeros(zeroN,1)];

K = 200;

F = category;

model = AsymNARM(B, F, K, xitrain, idtrain, xitest,idtest);
fprintf('AsymNARM, AUCroc =  %.4f, AUCpr = %.4f\n',model.AUCroc,model.AUCpr);

