
TrainX = load('C:/mydata_add/en_de_dvd_music/Train.data');
TrainX = spconvert(TrainX);
TrainY = load('C:/mydata_add/en_de_dvd_music/Train.label');
TrainY = TrainY';
TestX = load('C:/mydata_add/en_de_dvd_music/Test.data');
TestX = spconvert(TestX);
%%
TestY = load('C:/mydata_add/en_de_dvd_music/Test.label');
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
Gs = [];
for i = 1:length(TrainY)
    if TrainY(i) == 1
        Gs(i,1) = 1;
        Gs(i,2) = 0;
    else
        Gs(i,1) = 0;
        Gs(i,2) = 1;
    end
end

Gt = [];
for i = 1:length(TestY)
    if TrainY(i) == 1
        Gt(i,1) = 1;
        Gt(i,2) = 0;
    else
        Gt(i,1) = 0;
        Gt(i,2) = 1;
    end
end
Xs = TrainX;
Xt = TestX;

%%NMF
r = 50;
all = [TrainX TestX];
[m,n] = size(all);
Winit = abs(randn(m,r));
Hinit = abs(randn(r,n));
[W,H] = nmf(full(all),Winit,Hinit,0.0000000000001,25,8000);

for id=1:size(W,2)
    W(:,id) = W(:,id)/sum(W(:,id));
end
xlswrite(strcat('W.xls'),W);
xlswrite(strcat('H.xls'),H');