classdef neuron < handle
    % define a neuron
    properties
        weights
        parameter = [1 0];
        gradient % gradient of cost wrt weighted total input
    end
    
    methods
        function obj = neuron(weights)
            obj.weights = weights;
        end
            
        function output = feedforward(obj,input)
            numberOfSamples = size(input,2);
            netInput = obj.weights'*[input; ones(1,numberOfSamples)];
            output = obj.activation(netInput);
        end
        
        function output = activation(obj,netInput)
            % let us use logistic activation
            output = obj.sigmoid(netInput);
        end
    
        function output = sigmoid(obj,input)
            output = sigmf(input,obj.parameter);
        end
        
    end
    
end