classdef sgd < handle
    properties
        learningRate = .01;
        batchSize = 10;
    end
    
    methods
        function sgdStep(obj,data)
            % batchSize = 1
            % h_j^L = output of the jth neuron of the Lth layer
            % z_j^L = weighted total input to the jth neuron of the Lth layer
            % w_{ij}^L = weight of the connection between
                % jth neuron of the Lth layer and ith neuron of the {L-1}st layer
            % J = total cost on the batch
            % sigma = activation function for any neuron
            % delta_j^L = partial of J wrt h_j^L
            % fundamental eqns
            % input > output eqn
            % h_j^L = sigma(z_j^L)
            % output > input eqn 
            % z_j^L = sum_i w_{ji}^L h_i^{L-1}
            % change in weight > change in error eqn
            % partial J wrt w_{ji}^L = delta_j^L  sigma'(z_j^L) h_i^{L-1} 
            % delta^{L+1} > delta^L eqn
            % delta_j^L = sum_i delta_i^{L+1} sigma'(z_i^{L+1}) w_{ij}^{L+1}
            correctLabels = data(1,:);
            features = data(2:end,:);
            obj.feedforward(features();
            delta = 2*(obj.outputrealLabels);)
            for L = obj.depth:-1:1
                delta = obj.layers(L).sgdStep(delta);
            end
            
        end
        
        function output = sgdStep(obj,delta)
            % delta_j^L = partial of J wrt h_j^L
            % where L is the layer number of 'this' layer, ie obj
            % delta_j^{L-1} = sum_i delta_i^L+ sigma'(z_i^L) w_{ij}^L
            % this function computes delta^{L-1} in terms of delta^L
            previousWidth = size(obj.weightMatrix,1); % previous layer's width
            deltaExt = delta*ones(1,previousWidth);
            diffActivationExt = obj.activation.diff(obj.netInput)*ones(1,previousWidth);
            output = sum(deltaExt.*diffActivationExt.*obj.weightMatrix,1)';
            % weight update
            deltaExt2 = ones(previousWidth,1)*delta';
            diffActivationExt2 = ones(previousWidth,1)*obj.activation.diff(obj.netInput');
            inputExt = ones(obj.width,1)*obj.input;
            gradient = deltaExt2.*diffActivationExt2.*inputExt;
            obj.weightMatrix = obj.weightMatrix-obj.neuralNet.learningRate*gradient;
        end
        
    end
    
end