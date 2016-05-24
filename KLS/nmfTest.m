
TrainX = load('C:/NMTF/Train.data');
TrainX = spconvert(TrainX);
TrainY = load('C:/NMTF/Train.label');
TrainY = TrainY';
TestX = load('C:/NMTF/Test.data');
TestX = spconvert(TestX);
%%
TestY = load('C:/NMTF/Test.label');
TestY = TestY';

for id = 1:length(TrainY)
    if TrainY(id) == 2
        TrainY(id) = -1;
    end
end

for id = 1:length(TestY)
    if TestY(id) == 2
        TestY(id) = -1;
    end
end
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
% SS = ones(size(Xs,2),50);
% for i = 1:size(SS,1)
%     SS(i,:) = SS(i,:)/sum(SS(i,:));
% end
% HH = xlsread(strcat('H.xls'));
% [m,n] = size(HH');
% [S,G] = nmf(full(HH'),abs(randn(m,2)),abs(randn(2,n)),0.0000000000001,25,8000);
% xlswrite(strcat('S.xls'),S);
% xlswrite(strcat('G.xls'),[G',TrainY']);
% Gp = G';
% right = 0;
% for id=1:size(Gp,1)
%     if Gp(id,1) > Gp(id,2) && TrainY(id) == 1
%         right = right+1;
%     end
%     if Gp(id,1) < Gp(id,2) && TrainY(id) == -1
%         right = right+1;
%     end
% end
% [right/size(TrainX,2)]
% return ;

%%NMF
r = 50;
all = [TrainX,TestX];
[m,n] = size(all);
[m,n]
Winit = abs(randn(m,r));
Hinit = abs(randn(r,n));

[W,H] = nmf(full(all),Winit,Hinit,0.0000000000001,25,8000);
for id=1:size(W,2)
    W(:,id) = W(:,id)/sum(W(:,id));
end
xlswrite(strcat('W.xls'),W);
xlswrite(strcat('H.xls'),H');