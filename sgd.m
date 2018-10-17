classdef sgd < handle
    properties
        neuralNet
        probabilityLaw
        inputs
        netInputs
        hiddenOutputs
        % delta_j^L = partial of cost wrt output of the jth neuron
        % at layer L
        delta
        learningRate = .5;
        batchSize = 1;
    end
    
    methods
        function obj = sgd(neuralNet)
            obj.neuralNet = neuralNet;
            obj.probabilityLaw = neuralNet.trainingError.probabilityLaw;
        end
        
        function repeat_random_batch(obj,numberOfReps)
            for i = 1:numberOfReps
                obj.random_batch()
            end
            
        end
        
        function random_batch(obj)
            batch = obj.probabilityLaw.sample(obj.batchSize);
            obj.on_batch(batch)
        end
        
        function on_batch(obj,batch)
            % h_j^L = output of the jth neuron of the Lth layer
            % z_j^L = weighted total input to the jth neuron of the Lth layer
            % w_{ij}^L = weight of the connection between
                % jth neuron of the Lth layer and ith neuron of the {L-1}st layer
            % J = total cost on the batch
            % sigma = activation function for any neuron
            % delta_j^L = partial of J wrt h_j^L
            % %% fundamental eqns
            % % input > output eqn
            % h_j^L = sigma(z_j^L)
            % % output > input eqn
            % z_j^L = sum_i w_{ji}^L h_i^{L-1}
            % % change in weight > change in error eqn
            % partial J wrt w_{ji}^L = delta_j^L  sigma'(z_j^L) h_i^{L-1} 
            % % delta^{L+1} > delta^L eqn
            % delta_j^L = sum_i delta_i^{L+1} sigma'(z_i^{L+1}) w_{ij}^{L+1}
            correctLabels = batch(1,:);
            features = batch(2:end,:);
            [obj.inputs, obj.netInputs, obj.hiddenOutputs, output] = obj.neuralNet.feedforward(features);
            obj.delta = cell(1,obj.neuralNet.depth);
            obj.delta{obj.neuralNet.depth} = 2*(output-correctLabels);
            for layerNumber = obj.neuralNet.depth-1:-1:1
                % compute this layer's delta
                obj.delta_recurrence(layerNumber);
            end
            
            for layerNumber = obj.neuralNet.depth:-1:1
                obj.weight_update(layerNumber);
            end
            
        end
        
        function delta_recurrence(obj,layerNumber)
            % this function computes delta^layerNumber
            % in terms of delta^{HIGHER LAYERS}
            % width(1) is for inputDimension, so +1
            width = obj.neuralNet.widths(layerNumber+1);
            deltaRep = obj.delta{layerNumber+1}*ones(1,width);
            diffActivationRep = obj.neuralNet.layers{layerNumber+1}.activation.diff(obj.netInputs{layerNumber+1})*ones(1,width);
            croppedWeightMatrix = obj.neuralNet.weightMatrices{layerNumber+1}(1:end-1,:);
            obj.delta{layerNumber} = sum(deltaRep.*diffActivationRep.*croppedWeightMatrix',1)';
        end
        
        function weight_update(obj,layerNumber)
            % width(1) is for inputDimension, so NOT layerNumber-1 !
            % but including bias term we have +1
            previousWidth = obj.neuralNet.widths(layerNumber)+1;
            deltaRep = ones(previousWidth,1)*obj.delta{layerNumber}';
            diffActivationRep = ones(previousWidth,1)*obj.neuralNet.layers{layerNumber}.activation.diff(obj.netInputs{layerNumber}');
            % previous layer hidden outputs or bias
            if layerNumber ~= 1
                previousHiddenOutputsRep = [obj.hiddenOutputs{layerNumber-1}; 1]*ones(1,obj.neuralNet.widths(layerNumber+1));
            else % updating first layer weights prev hidden outputs are inputs
                previousHiddenOutputsRep = [obj.inputs; 1]*ones(1,obj.neuralNet.widths(layerNumber+1));
            end
            
            gradient = deltaRep.*diffActivationRep.*previousHiddenOutputsRep;
            obj.neuralNet.weightMatrices{layerNumber} = obj.neuralNet.weightMatrices{layerNumber}-obj.learningRate*gradient;
            obj.learning_rate_update();
        end
        
        function learning_rate_update(obj)
            t = 0.5/obj.learningRate;
            obj.learningRate = 0.5/(t+.01);
        end
        
    end
    
end