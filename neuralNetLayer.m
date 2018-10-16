classdef neuralNetLayer < handle
    % define a layer of neurons
    properties
        neuralNet
        layerNumber
        weightMatrix % include weight for bias term
        width
        neurons
        activation
    end
    
    methods
        function obj = neuralNetLayer(neuralNet,layerNumber)
            obj.neuralNet = neuralNet;
            obj.layerNumber = layerNumber;
            obj.weightMatrix = neuralNet.weightMatrices{layerNumber};
            obj.width = size(obj.weightMatrix,2);
            obj.neurons = cell(obj.width,1);
            for i = 1:obj.width
                obj.neurons{i} = neuron(obj,i);
            end
            
            obj.activation = activation();
        end
            
        function output = feedforward(obj,input)
            % size(input) = [inputDimension numberOfSamples]
            numberOfSamples = size(input,2);
            netInput = obj.weightMatrix'*[input; ones(1,numberOfSamples)];
            output = obj.activation.output(netInput);
        end
        
    end
        
end