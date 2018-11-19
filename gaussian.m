% compare 10 layer net with maximum likelihood and approximate maximum
% likelihood in terms of detection probability
cov = 0.5*eye(4);
train0 = mvnrnd(1/2*ones(1e4,4),cov);
test0 = mvnrnd(1/2*ones(1e4,4),cov);
%eps = 2^(-1);
cov(4:3:end-1) = eps;
train1 = mvnrnd(1/2*ones(1e4,4),cov);
test1 = mvnrnd(1/2*ones(1e4,4),cov);
train = [train0; train1];
train = reshape(train',2,2,1,2e4);
test = [test0; test1];
test = reshape(test',2,2,1,2e4);
class = zeros(2e4,1);
class(1e4+1:end) = 1;
layers = [
    imageInputLayer([2 2 1])

    convolution2dLayer(2,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(2,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer];
miniBatchSize = 128;
options = trainingOptions('sgdm', ...
'MiniBatchSize',miniBatchSize, ...
'MaxEpochs',10, ...
'InitialLearnRate',1e-3, ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.1, ...
'LearnRateDropPeriod',20, ...
'Shuffle','every-epoch', ...
'Plots','training-progress', ...
'Verbose',false);
net = trainNetwork(train,class,layers,options);
predictedClass = predict(net,test);
testError = sum(abs(predictedClass-class))/2e4
% for i = 1:10
%     subplot(2,10,i), imshow(train(:,:,:,i))
%     subplot(2,10,i+10), imshow(train(:,:,:,1e4+i))
% end

llr = (test(1,1,1,:)-.5).*(test(2,2,1,:)-.5)+(test(1,2,1,:)-.5).*(test(2,1,1,:)-.5); % approximate and scaled
mlPredictedClass = llr(:)>eps;
mlTestError = sum(abs(mlPredictedClass-class))/2e4
inv0 = 2*eye(4);
inv1 = inv(cov);
shift = .5*log(0.5^4/det(cov))
realMLPredicted = 1/2*ones(2e4,1);
for i = 1:20e3
    x = test(:,:,1,i);
    x = x(:);
    score = -0.5*(x-.5)'*(inv1-inv0)*(x-.5);
    realMLPredicted(i) = shift+score>0;
end

mlTestError = sum(abs(realMLPredicted-class))/2e4