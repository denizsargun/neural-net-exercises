classdef top_and_geo < handle
    % function approximation
    properties
        netSize
        nets
        noParam
        regPen
        data
        thr
    end
    
    methods
        function obj = top_and_geo()
            obj.netSize = [1; 2; 1];
            net = feedforwardnet(obj.netSize(2:end-1));
            net.inputs{1}.size = 1;
            net.layers{end}.size = 1;
            obj.regPen = 0.05;
            [feat, tar] = obj.generate_data(1e7);
            obj.data = {feat; tar};
            % threshold for which network satisfies e_loss < thr
            % for a normal vector with sigma_X,sigma_Y = 1, rho = .5, ...
            % min error=sigma_Y^2(1-rho^2)=.75
            obj.thr = .93;
            net.trainParam.goal = obj.thr;
            % show/not show training window
            net.trainParam.showWindow = false;
            % regularization sets the performance as the weighted average
            % of mean squared error and L2 norm of weigths and biases
            % perf = regParam*L2+(1-regParam)*mse
            net.performParam.regularization = 0.05;
            obj.nets{1} = init(net);
            obj.nets{2} = init(net);
        end
        
        function [feat, tar] = generate_data(obj,numberOfSamples)
            % function approximation task, y = 1/2*x
            d = mvnrnd(zeros(numberOfSamples,2),[1 .5; .5 1])';
            feat = d(1,:);
            tar = d(2,:);
        end
        
        % oracle loss
        function loss = o_loss(obj,netIndex)
            net = obj.nets{netIndex};
            numberOfSamples = 1e6;
            [feat, tar] = obj.generate_data(numberOfSamples);
            empLoss = 1/numberOfSamples*norm(net(feat)-tar,2)^2;
            regLoss = sum(sum(abs(net.IW{1}))) ...
                +sum(sum(abs(net.IW{2}))) ...
                +sum(sum(abs(net.LW{1}))) ...
                +sum(sum(abs(net.LW{2}))) ...
                +sum(abs(net.b{1}))+sum(abs(net.b{2}));
            loss = (1-obj.regPen)*empLoss+obj.regPen*regLoss;
        end
        
        % empirical loss
        function loss = e_loss(obj,netIndex)
            net = obj.nets{netIndex};
            numberOfSamples = length(obj.data{2});
            empLoss = 1/numberOfSamples ...
                *norm(net(obj.data{1})-obj.data{2},2)^2;
            % L1 loss
            regLoss = sum(sum(abs(net.IW{1}))) ...
                +sum(sum(abs(net.IW{2}))) ...
                +sum(sum(abs(net.LW{1}))) ...
                +sum(sum(abs(net.LW{2}))) ...
                +sum(abs(net.b{1}))+sum(abs(net.b{2}));
            loss = (1-obj.regPen)*empLoss+obj.regPen*regLoss;
        end
        
        function train(obj,netIndex)
            % deeper network options
            % options = trainingOptions('sgdm', ...
            % 'ValidationFrequency', 1e6, ...
            % 'MiniBatchSize',16, ...
            % 'L2Regularization',0.001);
            [feat, tar] = generate_data(obj,1e6);
            obj.nets{netIndex} = train(obj.nets{netIndex},feat,tar);
        end
        
        function pick_two(obj)
            obj.train(1);
            obj.train(2);
        end
        
        function pick_good_two(obj)
            it = 1;
            lossThr = 0;
            while ~lossThr
                net = init(obj.nets{1});
%                 net.IW{1} = (obj.nets{index1}.IW{1}+obj.nets{index2}.IW{1})/2;
%                 net.LW{2,1} = (obj.nets{index1}.LW{2,1} ...
%                     +obj.nets{index2}.LW{2,1})/2;
%                 net.b{1} = (obj.nets{index1}.b{1}+obj.nets{index2}.b{1})/2;
%                 net.b{2} = (obj.nets{index1}.b{2}+obj.nets{index2}.b{2})/2;
                obj.nets{1} = net;
                obj.train(1)
                loss = obj.e_loss(1);
                lossThr = loss <= obj.thr;
                it = it+1;
                if it > 1e3
                    break
                end
                
            end
            
            it = 1;
            lossThr = 0;
            while ~lossThr
                net = init(obj.nets{2});
%                 net.IW{1} = (obj.nets{index1}.IW{1}+obj.nets{index2}.IW{1})/2;
%                 net.LW{2,1} = (obj.nets{index1}.LW{2,1} ...
%                     +obj.nets{index2}.LW{2,1})/2;
%                 net.b{1} = (obj.nets{index1}.b{1}+obj.nets{index2}.b{1})/2;
%                 net.b{2} = (obj.nets{index1}.b{2}+obj.nets{index2}.b{2})/2;
                obj.nets{2} = net;
                obj.train(2)
                loss = obj.e_loss(2);
                lossThr = loss <= obj.thr;
                it = it+1;
                if it > 1e3
                    break
                end
                
            end
            
        end
        
        function train_mid_net(obj,index1,index2)
            % initialize
            newIndex = length(obj.nets)+1;
            % inRange = 0;
            isCloser = 0;
            lossThr = 0;
            it = 1;
            while (~isCloser||~lossThr)&&it<=1000
                net = init(obj.nets{index1});
                net.IW{1} = ...
                    (obj.nets{index1}.IW{1}-obj.nets{index2}.IW{1}) ...
                    .*[rand; rand]+obj.nets{index2}.IW{1};
                net.LW{2,1} = (obj.nets{index1}.LW{2,1} ...
                    -obj.nets{index2}.LW{2,1}).*[rand rand] ...
                    +obj.nets{index2}.LW{2,1};
                net.b{1} = ...
                    (obj.nets{index1}.b{1}-obj.nets{index2}.b{1}) ...
                    .*[rand; rand]+obj.nets{index2}.b{1};
                net.b{2} = (obj.nets{index1}.b{2} ...
                    -obj.nets{index2}.b{2})*rand+obj.nets{index2}.b{2};
                obj.nets{newIndex} = net;
                obj.train(newIndex)
                % inRange and isCloser are two different conditions
                % inRange = obj.is_net_param_in_range(index1,index2,newIndex);
                isCloser = obj.is_net_param_closer_to_both(index1,index2,newIndex);
                loss = obj.e_loss(newIndex);
                lossThr = loss <= obj.thr;
                it = it+1;
            end
            
        end
        
        function inRange = is_net_param_in_range(obj,index1,index2,midIndex)
            param1 = obj.nets{index1}.IW{1}(:);
            param1 = [param1; obj.nets{index1}.LW{2,1}(:)];
            param1 = [param1; obj.nets{index1}.b{1}(:)];
            param1 = [param1; obj.nets{index1}.b{2}(:)];
            param2 = obj.nets{index2}.IW{1}(:);
            param2 = [param2; obj.nets{index2}.LW{2,1}(:)];
            param2 = [param2; obj.nets{index2}.b{1}(:)];
            param2 = [param2; obj.nets{index2}.b{2}(:)];
            paramMid = obj.nets{midIndex}.IW{1}(:);
            paramMid = [paramMid; obj.nets{midIndex}.LW{2,1}(:)];
            paramMid = [paramMid; obj.nets{midIndex}.b{1}(:)];
            paramMid = [paramMid; obj.nets{midIndex}.b{2}(:)];
            inRange = (sum((paramMid>=min(param1,param2))==0)==0) ...
                *(sum((paramMid<=max(param1,param2))==0)==0);
        end
        
        function isCloser = is_net_param_closer_to_both(obj,index1,index2,midIndex)
            param1 = obj.nets{index1}.IW{1}(:);
            param1 = [param1; obj.nets{index1}.LW{2,1}(:)];
            param1 = [param1; obj.nets{index1}.b{1}(:)];
            param1 = [param1; obj.nets{index1}.b{2}(:)];
            obj.noParam = length(param1);
            param2 = obj.nets{index2}.IW{1}(:);
            param2 = [param2; obj.nets{index2}.LW{2,1}(:)];
            param2 = [param2; obj.nets{index2}.b{1}(:)];
            param2 = [param2; obj.nets{index2}.b{2}(:)];
            paramMid = obj.nets{midIndex}.IW{1}(:);
            paramMid = [paramMid; obj.nets{midIndex}.LW{2,1}(:)];
            paramMid = [paramMid; obj.nets{midIndex}.b{1}(:)];
            paramMid = [paramMid; obj.nets{midIndex}.b{2}(:)];
            dist12 = norm(param1-param2);
            dist13 = norm(param1-paramMid);
            dist23 = norm(param2-paramMid);
            isCloser = max(dist13,dist23)<dist12;
        end
        
        function dist_matrix(obj)
            noNets = length(obj.nets);
            matrix = zeros(noNets);
            for i = 1:noNets-1
                for j = i+1:noNets
                    param1 = obj.nets{i}.IW{1}(:);
                    param1 = [param1; obj.nets{i}.LW{2,1}(:)];
                    param1 = [param1; obj.nets{i}.b{1}(:)];
                    param1 = [param1; obj.nets{i}.b{2}(:)];
                    param2 = obj.nets{j}.IW{1}(:);
                    param2 = [param2; obj.nets{j}.LW{2,1}(:)];
                    param2 = [param2; obj.nets{j}.b{1}(:)];
                    param2 = [param2; obj.nets{j}.b{2}(:)];
                    matrix(i,j) = norm(param1-param2);
                end
                
            end
            
        end
        
    end
    
    methods (Static)
    end
    
end