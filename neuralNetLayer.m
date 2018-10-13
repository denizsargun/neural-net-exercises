classdef neuralNetLayer < handle
    % define a layer of neurons
    properties
        width
        neurons
        input
        output
    end
    
    methods
        function obj = neuralNetLayer(weightMatrix)
            obj.width = size(weightMatrix,2);
            for i = 1:obj.width
                neurons(i,1) = neuron(weightMatrix(:,i));
            end
            
            obj.neurons = neurons;
        end
            
        function feedforward(obj,input)
            obj.input = input;
            obj.output = zeros(obj.width,1);
            for i = 1:obj.width
                obj.neurons(i).feedforward(input);
                obj.output(i) = obj.neurons(i).output;
            end
            
        end
        
    end
        
end