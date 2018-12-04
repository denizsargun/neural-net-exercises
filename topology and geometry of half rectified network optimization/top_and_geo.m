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
            obj.nn = feedforwardnet(obj.netSize(2:end-1));
            [feat, tar] = obj.generate_data(2);
            obj.nn = configure(obj.nn,feat,tar);
            obj.nn = init(obj.nn);
            obj.regPen = 1e-3;
            [feat, tar] = obj.generate_data(1e7);
            obj.data = {feat; tar};
            % threshold for which network satisfies e_loss < thr
            obj.thr = 1e-2;
        end
        
        function [feat, tar] = generate_data(obj,numberOfSamples)
            % function approximation task, y = 1/2*x
            d = mvnrnd(zeros(numberOfSamples,2),[1 .5; .5 1])';
            feat = d(:,1);
            tar = d(:,2);
        end
        
        % oracle loss
        function loss = o_loss(obj)
            numberOfSamples = 1e6;
            [feat, tar] = obj.generate_data(numberOfSamples);
            empLoss = 1/numberOfSamples*norm(obj.nn(feat)-tar,2);
            regLoss = sum(sum(abs(obj.nn.IW{1}))) ...
                +sum(sum(abs(obj.nn.IW{2}))) ...
                +sum(sum(abs(obj.nn.LW{1}))) ...
                +sum(sum(abs(obj.nn.LW{2}))) ...
                +sum(abs(obj.nn.b{1}))+sum(abs(obj.nn.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        function loss = o_loss2(obj)
            numberOfSamples = 1e6;
            [feat, tar] = obj.generate_data(numberOfSamples);
            empLoss = 1/numberOfSamples*norm(obj.nn2(feat)-tar,2);
            regLoss = sum(sum(abs(obj.nn2.IW{1}))) ...
                +sum(sum(abs(obj.nn2.IW{2}))) ...
                +sum(sum(abs(obj.nn2.LW{1}))) ...
                +sum(sum(abs(obj.nn2.LW{2}))) ...
                +sum(abs(obj.nn2.b{1}))+sum(abs(obj.nn2.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        % empirical loss
        function loss = e_loss(obj)
            numberOfSamples = length(obj.data{2});
            empLoss = 1/numberOfSamples*norm(obj.nn(obj.data{1})-obj.data{2},2);
            regLoss = sum(sum(abs(obj.nn.IW{1}))) ...
                +sum(sum(abs(obj.nn.IW{2}))) ...
                +sum(sum(abs(obj.nn.LW{1}))) ...
                +sum(sum(abs(obj.nn.LW{2}))) ...
                +sum(abs(obj.nn.b{1}))+sum(abs(obj.nn.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        function loss = e_loss2(obj)
            numberOfSamples = length(obj.data{2});
            empLoss = 1/numberOfSamples*norm(obj.nn2(obj.data{1})-obj.data{2},2);
            regLoss = sum(sum(abs(obj.nn2.IW{1}))) ...
                +sum(sum(abs(obj.nn2.IW{2}))) ...
                +sum(sum(abs(obj.nn2.LW{1}))) ...
                +sum(sum(abs(obj.nn2.LW{2}))) ...
                +sum(abs(obj.nn2.b{1}))+sum(abs(obj.nn2.b{2}));
            loss = empLoss+obj.regPen*regLoss;
        end
        
        function train(obj)
            [feat, tar] = generate_data(obj,1e6);
            % deeper network options
%             options = trainingOptions('sgdm', ...
%                 'ValidationFrequency', 1e6, ...
%                 'MiniBatchSize',16, ...
%                 'L2Regularization',0.001);
            obj.nn = train(obj.nn,feat,tar);
        end
        
        function train2(obj)
            [feat, tar] = generate_data(obj,1e6);
            % deeper network options
%             options = trainingOptions('sgdm', ...
%                 'ValidationFrequency', 1e6, ...
%                 'MiniBatchSize',16, ...
%                 'L2Regularization',0.001);
            obj.nn2 = train(obj.nn2,feat,tar);
        end
        
        function pick_two(obj)
            obj.nn = init(obj.nn);
            obj.nn2 = obj.nn;
            obj.train;
            obj.e_loss()
            obj.o_loss()
            obj.train2;
            obj.e_loss2()
            obj.o_loss2()
        end
        
    end
    
    methods (Static)
    end
    
end