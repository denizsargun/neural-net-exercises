classdef neuralNet < handle
    % define a neural net
    properties
        probabilityLaw % aim to learn this dist by sampling
        depth
        weightMatrices % cell array
        layers
        batchSize = 10;
    end
    
    methods
        function obj = neuralNet(weightMatrices)
            obj.probabilityLaw = probabilityLaw();
            obj.depth = length(weightMatrices);
            obj.weightMatrices = weightMatrices;
            for i = 1:obj.depth
                layers(i,1) = neuralNetLayer(weightMatrices{i});
            end
            
            obj.layers = layers;
        end
            
        function output = feedforward(obj,input)
            % input of size inputDimension x numberOfSamples
            % output of size outputDimension x numberOfSamples
            for i = 1:obj.depth
                output = obj.layers(i).feedforward(input);
                input = output;
            end
            
        end
        
        function output = error(obj,data)
            % data = [correct label; features]
            correctLabel = data(1,:);
            input = data(2:end,:);
            classEstimate = obj.feedforward(input);
            output = obj.lossFunction(classEstimate-correctLabel);
        end
        
        function output = getBatch(obj,batchSize)
            output = obj.probabilityLaw.sample(batchSize);
        end
        
        function sgdStep(obj,batch)
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
            for L = obj.depth:-1:1
                delta = obj.layers(L).sgdStep(delta,batch);
            end
            
        end
        
    end
    
    methods (Static)
        function output = lossFunction(difference)
            % L2 loss
            output = norm(difference);
        end
        
    end
        
end