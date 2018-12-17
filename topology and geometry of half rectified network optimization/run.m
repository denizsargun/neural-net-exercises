A = top_and_geo;
A.pick_good_two()
A.train_mid_net(1,2)
A.train_mid_net(1,3)
A.train_mid_net(3,2)
A.train_mid_net(1,4)
A.train_mid_net(3,4)
A.train_mid_net(3,5)
A.train_mid_net(5,2)
A.train_mid_net(1,6)
A.train_mid_net(4,6)
A.train_mid_net(4,7)
A.train_mid_net(3,7)
A.train_mid_net(3,8)
A.train_mid_net(5,8)
A.train_mid_net(5,9)
A.train_mid_net(2,9)
paramMat = zeros(A.noParam,length(A.nets));
for i = 1:length(A.nets)
    param = A.nets{i}.IW{1}(:);
    param = [param; A.nets{i}.LW{2,1}(:)];
    param = [param; A.nets{i}.b{1}(:)];
    param = [param; A.nets{i}.b{2}(:)];
    paramMat(:,i) = param;
end

[U,S,V] = svd(paramMat);
% reduce dimension to 2
% plot loss surface
res = 20;
eLoss = zeros(20);
coor = S*V';
redCoor = coor(1:2,:);
firstCoorRan = [min(redCoor(1,:));max(redCoor(1,:))];
diff1 = (firstCoorRan(2)-firstCoorRan(1))/(res-1);
secondCoorRan = [min(redCoor(2,:));max(redCoor(2,:))];
diff2 = (secondCoorRan(2)-secondCoorRan(1))/(res-1);
coor1 = firstCoorRan(1):diff1:firstCoorRan(2);
coor2 = secondCoorRan(1):diff2:secondCoorRan(2);
dummyNetIndex = length(A.nets)+1;
for i = 1:res
    for j = 1:res
        myCoor1 = coor1(i);
        myCoor2 = coor2(j);
        netParam = U(:,1:2)*[myCoor1; myCoor2];
        net = feedforwardnet(A.netSize(2:end-1));
        net.inputs{1}.size = 1;
        net.layers{end}.size = 1;
        net.IW{1} = netParam(1:2);
        net.LW{2,1} = netParam(3:4)';
        net.b{1} = netParam(5:6);
        net.b{2} = netParam(7);
        A.nets{dummyNetIndex} = net;
        eLoss(i,j) = A.e_loss(dummyNetIndex);
    end
    
end