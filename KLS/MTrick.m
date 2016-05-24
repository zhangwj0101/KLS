function Results = MTrick(TrainX,TrainY,TestX,TestY,alpha,beta,numK,numCircle)
%%% 联合训练 列为一
G0 = [];
for i = 1:length(TrainY)
    if TrainY(i) == 1
        G0(i,1) = 1;
        G0(i,2) = 0;
    else
        G0(i,1) = 0;
        G0(i,2) = 1;
    end
end

%%%逻辑回归
TrainXY = scale_cols(TrainX,TrainY);
fprintf('......start to train logistic regression model.........\n');
w00 = zeros(size(TrainXY,1),1);
lambda = exp(linspace(-0.5,6,20));
wbest = [];
f1max = -inf;
for i = 1:length(lambda)
    w_0 = train_cg(TrainXY,w00,lambda(i));
    f1 = logProb(TrainXY,w_0);
    if f1 > f1max
        f1max = f1;
        wbest = w_0;
        se_lambda = lambda(i);
    end
end
ptemp = 1./(1 + exp(-wbest'*TrainX));
oriA = getResult(ptemp,TrainY);
fprintf('Test accuracy on source domain is :%g\n',oriA);
ptemp = 1./(1 + exp(-wbest'*TestX));

oriA = getResult(ptemp,TestY);
fprintf('Test accuracy on target domain is :%g\n',oriA);
Gt = [];
for i = 1:length(TestY)
    Gt(i,1) = ptemp(i);
    Gt(i,2) = 1 - ptemp(i);
end
%%%%逻辑回归结束
%%%NMF way
r = numK;
all = [TrainX TestX];
[m,n] = size(all);
Winit = abs(randn(m,r));
Hinit = abs(randn(r,n));
[W,H] = nmf(full(all),Winit,Hinit,0.0000000000001,25,8000);

for id=1:size(W,2)
    W(:,id) = W(:,id)/sum(W(:,id));
end
%%%%end NMF way
Xs = TrainX;
Xt = TestX;

for i = 1:size(TrainX,2)
    Xs(:,i) = Xs(:,i)/sum(Xs(:,i));
end
for i = 1:size(TestX,2)
    Xt(:,i) = Xt(:,i)/sum(Xt(:,i));
end
XX = [Xs Xt];
FF = W;
% SS = S;

SS = ones(numK,2);
for i = 1:size(SS,1)
    SS(i,:) = SS(i,:)/sum(SS(i,:));
end
GG = [G0;Gt];

for circleID = 1:numCircle
    %%FF
    tempM = (FF*SS*GG'*GG*SS');
    tempM1 = XX*GG*SS';
    for i = 1:size(FF,1)
        for j = 1:size(FF,2)
            if tempM(i,j)~=0
                FF(i,j) = FF(i,j)*(tempM1(i,j)/tempM(i,j))^(0.5);
            else
                FF(i,j) = 0;
            end
        end
    end

    for i = 1:size(FF,2)
        if sum(FF(:,i))~= 0
            FF(:,i) = FF(:,i)/sum(FF(:,i));
        else
            for j = 1:size(FF,2)
                FF(i,j) = 1/(size(FF,2));
            end
        end
    end
    %%SS
    tempM = (FF'*FF*SS*GG'*GG);
    tempM1 = FF'*XX*GG;
    for i = 1:size(SS,1)
        for j = 1:size(SS,2)
            if tempM(i,j)~=0
                SS(i,j) = SS(i,j)*(tempM1(i,j)/tempM(i,j))^(0.5);
            else
                SS(i,j) = 0;
            end
        end
    end
   
    %% Gt
    tempM = (GG*SS'*FF'*FF*SS);
    tempM1 = XX'*FF*SS;
    for i = 1:size(GG,1)
        for j = 1:size(GG,2)
            if tempM(i,j)~=0
                GG(i,j) = GG(i,j)*(tempM1(i,j)/tempM(i,j))^(0.5);
            else
                GG(i,j) = 0;
            end
        end
    end
    
    for i = 1:size(GG,1)
        if sum(GG(i,:))~= 0
            GG(i,:) = GG(i,:)/sum(GG(i,:));
        else
            for j = 1:size(GG,2)
                GG(i,j) = 1/(size(GG,2));
            end
        end
    end
    
    for i = 1:size(G0,1)
        GG(i,:) = G0(i,:);
    end
  
    pp = [];
    for i = 1:length(TestY)
        idx  = size(G0,1)+i;
        if sum(GG(idx,:))~= 0
            pp(1,i) = GG(idx,1)/sum(GG(idx,:));
        else
            pp(1,i) = 0.5;
        end
    end
    Results(circleID) = getResult(pp,TestY)*100;
    %     lvalues(circleID) = trace(Ft'*Ft-2*Ft'*Fs+Fs'*Fs);
    
    fprintf('the %g iteration is %g, the max is %g. \n',circleID,getResult(pp,TestY),max(Results));
end
