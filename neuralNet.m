classdef neuralNet < handle
    % define a neural net
    properties
        weightMatrices % cell array
        depth
        layers
        trainingError
    end
    
    methods
        function obj = neuralNet(weightMatrices)
            obj.weightMatrices = weightMatrices;
            obj.depth = length(weightMatrices);
            obj.layers = cell(1,obj.depth);
            for i = 1:obj.depth
                obj.layers{i} = neuralNetLayer(obj,i);
            end
            
            obj.trainingError = trainingError(obj);
        end
            
        function [hidden, output] = feedforward(obj,input)
            % size(input) = [inputDimension numberOfSamples]
            % size(output) = [outputDimension numberOfSamples]
            hidden = cell(1,obj.depth);
            for i = 1:obj.depth
                hidden{i} = obj.layers(i).feedforward(input);
                input = hidden{i};
            end
            
            output = hidden{end};
        end
        
    end
        
end