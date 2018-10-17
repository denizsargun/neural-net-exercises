classdef neuron < handle
    % define a neuron
    properties
        layer
        neuronNumber
        % define weights over the neural net
        % weights
        activation
    end
    
    methods
        function obj = neuron(layer,neuronNumber)
            obj.layer = layer;
            obj.neuronNumber = neuronNumber;
            obj.activation = activation();
        end
            
        function output = feedforward(obj,inputs)
            % size(inputs) = [inputDimension numberOfSamples]
            numberOfSamples = size(inputs,2);
            weights = obj.layer.neuralNet.weightMatrices{obj.layer.layerNumber}{obj.neuronNumber};
            netInput = weights'*[inputs; ones(1,numberOfSamples)];
            output = obj.activation.output(netInput);
        end
        
    end
    
end