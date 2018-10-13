classdef neuralNet < handle
    % define a neural net
    properties
        numberOfLayers
        layers
        input
        output
    end
    
    methods
        function obj = neuralNet(weightMatrices)
            obj.numberOfLayers = size(weightMatrices,3);
            for i = 1:obj.numberOfLayers
                layers(i,1) = neuralNetLayer(weightMatrices(:,:,i));
            end
            
            obj.layers = layers;
        end
            
        function feedforward(obj,input)
            obj.input = input;
            for i = 1:obj.numberOfLayers
                obj.layers(i).feedforward(input);
                input = obj.layers(i).output;
            end
            
            obj.output = obj.layers(end).output;
        end
        
    end
        
end