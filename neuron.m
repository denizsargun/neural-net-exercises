classdef neuron < handle
    % define a neuron
    properties
        weights
        activation
        gradient % gradient of cost wrt weighted total input
    end
    
    methods
        function obj = neuron(weights)
            obj.weights = weights;
            obj.activation = activation();
        end
            
        function output = feedforward(obj,input)
            numberOfSamples = size(input,2);
            netInput = obj.weights'*[input; ones(1,numberOfSamples)];
            output = obj.activation.output(netInput);
        end
        
    end
    
end