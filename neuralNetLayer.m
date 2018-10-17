classdef neuralNetLayer < handle
    % define a layer of neurons
    properties
        neuralNet
        layerNumber
        % define weightMatrix over the neural net
        % weightMatrix % include weight for bias term
        width
        neurons
        activation
    end
    
    methods
        function obj = neuralNetLayer(neuralNet,layerNumber)
            obj.neuralNet = neuralNet;
            obj.layerNumber = layerNumber;
            obj.width = neuralNet.widths(layerNumber+1);
            obj.neurons = cell(obj.width,1);
            for i = 1:obj.width
                obj.neurons{i} = neuron(obj,i);
            end
            
            obj.activation = activation();
        end
            
        function [netInputs, outputs] = feedforward(obj,inputs)
            % size(inputs) = [inputDimension numberOfSamples]
            numberOfSamples = size(inputs,2);
            weightMatrix = obj.neuralNet.weightMatrices{obj.layerNumber};
            netInputs = weightMatrix'*[inputs; ones(1,numberOfSamples)];
            outputs = obj.activation.output(netInputs);
        end
        
    end
        
end