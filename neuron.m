classdef neuron < handle
    % define a neuron
    properties
        weights
        input
        netInput
        output
    end
    
    methods
        function obj = neuron(weights)
            obj.weights = weights;
        end
            
        function feedforward(obj,input)
            obj.input = input;
            obj.netInput = obj.weights'*[input; 1];
            obj.activation()
        end
        
        function activation(obj)
            % let us use logistic activation
            obj.output = obj.sigmoid(obj.netInput);
        end
                
    end
    
    methods (Static)
        function output = sigmoid(input)
            parameter = [1 0];
            output = sigmf(input,parameter);
        end
        
    end
    
end