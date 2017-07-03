function [xitrain, idtrain, xitest,idtest]=datasplit(xi,id,trainfrac)

    testfrac = 1 - trainfrac;
    N=length(xi);
    Ntest=floor(N*testfrac);
    Ntrain=N-Ntest;
    % rand('state',0);%generate same randome numbers each time
    IndPerm=randperm(N);
    idtrain=zeros(Ntrain,2);
    xitrain=zeros(Ntrain,1);
    idtest=zeros(Ntest,2);
    xitest=zeros(Ntest,1);
    for i=1:N
        if i<=Ntrain
            xitrain(i)=xi(IndPerm(i));
            idtrain(i,:)=id(IndPerm(i),:);
        else
            xitest(i-Ntrain)=xi(IndPerm(i));
            idtest(i-Ntrain,:)=id(IndPerm(i),:);
        end
    end
end