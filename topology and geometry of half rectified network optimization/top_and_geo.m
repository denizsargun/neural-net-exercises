classdef top_and_geo < handle
    % function approximation
    properties
        netSize
        nn
        nn2
        regPen
        data
        thr
    end
    
    methods
        function obj = top_and_geo()
            obj.netSize = [1; 2; 1];
            nn = feedforwardnet(obj.netSize(2:end-1));
            nn.inputs{1}.size = 1;
            nn.layers{end}.size = 1;
            obj.nn = init(nn);
            obj.nn2 = init(nn);
            obj.regPen = 1e-3;
            [feat, tar] = obj.generate_data(1e7);
            obj.data = {feat; tar};
            % threshold for which network satisfies e_loss < thr
            obj.thr = 1e-2;
        end
        
        function [feat, tar] = generate_data(obj,numberOfSamples)
            % function approximation task, y = 1/2*x
            d = mvnrnd(zeros(numberOfSamples,2),[1 .5; .5 1])';
            feat = d(1,:);
            tar = d(2,:);
        end
        
        % oracle loss
        function loss = o_loss(obj,netIndex)
            if netIndex == 1
                net = obj.nn;
            elseif netIndex == 2
                net = obj.nn2;
            end
            
            numberOfSamples = 1e6;
            [feat, tar] = obj.generate_data(numberOfSamples);
            empLoss = 1/numberOfSamples*norm(net(feat)-tar,2);
            regLoss = sum(sum(abs(net.IW{1}))) ...
                +sum(sum(abs(net.IW{2}))) ...
                +sum(sum(abs(net.LW{1}))) ...
                +sum(sum(abs(net.LW{2}))) ...
                +sum(abs(net.b{1}))+sum(abs(net.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        % empirical loss
        function loss = e_loss(obj,netIndex)
            if netIndex == 1
                net = obj.nn;
            elseif netIndex == 2
                net = obj.nn2;
            end
            
            numberOfSamples = length(obj.data{2});
            empLoss = 1/numberOfSamples*norm(net(obj.data{1})-obj.data{2},2);
            regLoss = sum(sum(abs(net.IW{1}))) ...
                +sum(sum(abs(net.IW{2}))) ...
                +sum(sum(abs(net.LW{1}))) ...
                +sum(sum(abs(net.LW{2}))) ...
                +sum(abs(net.b{1}))+sum(abs(net.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        function train(obj,netIndex)
            % deeper network options
%             options = trainingOptions('sgdm', ...
%                 'ValidationFrequency', 1e6, ...
%                 'MiniBatchSize',16, ...
%                 'L2Regularization',0.001);
            [feat, tar] = generate_data(obj,1e6);
            if netIndex == 1
                obj.nn = train(obj.nn,feat,tar);
            elseif netIndex == 2
                obj.nn2 = train(obj.nn2,feat,tar);
            end
            
        end
        
        function pick_two(obj)
            1
            obj.e_loss(1)
            obj.o_loss(1)
            obj.train(1);
            obj.e_loss(1)
            obj.o_loss(1)
            2
            obj.e_loss(2)
            obj.o_loss(2)
            obj.train(2);
            obj.e_loss(2)
            obj.o_loss(2)
        end
        
    end
    
    methods (Static)
    end
    
end