classdef neuron < handle
    % define a neuron
    properties
        layer
        neuronNumber
        weights
        activation
    end
    
    methods
        function obj = neuron(layer,neuronNumber)
            obj.layer = layer;
            obj.neuronNumber = neuronNumber;
            obj.weights = obj.layer.weightMatrix(:,neuronNumber);
            obj.activation = activation();
        end
            
        function output = feedforward(obj,input)
            % size(input) = [inputDimension numberOfSamples]
            numberOfSamples = size(input,2);
            netInput = obj.weights'*[input; ones(1,numberOfSamples)];
            output = obj.activation.output(netInput);
        end
        
    end
    
end