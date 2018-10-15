classdef neuralNetLayer < handle
    % define a layer of neurons
    properties
        neuralNet
        weightMatrix % include weight for bias term
        width
        neurons
        activation
        % input and output is required for sgdStep
        input
        netInput
        output
    end
    
    methods
        function obj = neuralNetLayer(neuralNet,weightMatrix)
            obj.neuralNet = neuralNet;
            obj.weightMatrix = weightMatrix;
            obj.width = size(weightMatrix,2);
            for i = 1:obj.width
                neurons(i,1) = neuron(weightMatrix(:,i));
            end
            
            obj.neurons = neurons;
            obj.activation = activation();
        end
            
        function output = feedforward(obj,input)
            obj.input = input;
            numberOfSamples = size(input,2);
            obj.netInput = obj.weightMatrix'*[input; ones(1,numberOfSamples)];
            % neuron-by-neuron output
%             numberOfSamples = size(input,2);
%             output = zeros(obj.width,numberOfSamples);
%             for i = 1:obj.width
%                 output(i,:) = obj.neurons(i).feedforward(input);
%             end
            
            output = obj.activation(obj.netInput);
            obj.output = output;
        end
        
        function output = sgdStep(delta)
            % delta_j^L = partial of J wrt h_j^L
            % where L is the layer number of 'this' layer, ie obj
            % delta_j^{L-1} = sum_i delta_i^L+ sigma'(z_i^L) w_{ij}^L
            % this function computes delta^{L-1} in terms of delta^L
            previousWidth = size(obj.weigthMatrix,1); % previous layer's width
            deltaExt = delta*ones(1,previousWidth);
            diffActivation = obj.activation.diff(obj.netInput)*ones(1,previousWidth);
            output = sum(deltaExt.*diffActivation.*obj.weightMatrix,1)';
            % weight update
            deltaExt2 = ones()*delta';
        end
        
    end
        
end