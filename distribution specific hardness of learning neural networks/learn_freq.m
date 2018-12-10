% frequency
f = 1;
x = rand(sampleSize,1);
y = sin(f*x);
% net width
width = 10;
net = feedforwardnet([width; 1]);
train(net,x,y)