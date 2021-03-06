%% read and train MNIST
%% version 1
clear
clc
path = '\\home2.coeit.osu.edu\s\sargun.1\ECE\Desktop\MNIST data\mnist\';
file = 'train-images.idx3-ubyte';
fileId = fopen(strcat(path,file));
rowSize = 28;
columnSize = 28;
sampleSize = 60000;
rawImgDataTrain = uint8(fread(fileId,rowSize*columnSize*sampleSize,'uint8'));
fclose(fileId);
rawImgDataTrain = reshape(rawImgDataTrain, [rowSize,columnSize,sampleSize]);
imgDataTrain = zeros(rowSize,columnSize,1,sampleSize);
for i = 1:sampleSize
    imgDataTrain(:,:,1,i) = uint8(rawImgDataTrain(:,:,i));
end

%% version 2

% Matlab R2016b does not allow to chose training on cpu as an option and
% the lab computer does not have a supported gpu, so we omit this case

% tweaked for Matlab R2017a
clear
clc

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 750;

[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% outputSize = (inputSize-filterSize)/stride+1
layers = [
imageInputLayer([28 28 1])

convolution2dLayer(3,8,'Padding',[1 3])
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,16,'Padding',[1 7])
reluLayer
maxPooling2dLayer(2,'Stride',2)

convolution2dLayer(3,32,'Padding',[1 7])
reluLayer

fullyConnectedLayer(10)
softmaxLayer

classificationLayer];

options = trainingOptions('sgdm', ...
'InitialLearnRate',0.01, ...
'MaxEpochs',4, ...
'Shuffle','once', ...
'Verbose',false);

[net, info] = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)