classdef neuralNetLayer < handle
    % define a layer of neurons
    properties
        weightMatrix
        width
        neurons
    end
    
    methods
        function obj = neuralNetLayer(weightMatrix)
            obj.weightMatrix = weightMatrix;
            obj.width = size(weightMatrix,2);
            for i = 1:obj.width
                neurons(i,1) = neuron(weightMatrix(:,i));
            end
            
            obj.neurons = neurons;
        end
            
        function output = feedforward(obj,input)
            numberOfSamples = size(input,2);
            output = zeros(obj.width,numberOfSamples);
            for i = 1:obj.width
                output(i,:) = obj.neurons(i).feedforward(input);
            end
            
        end
        
    end
        
end