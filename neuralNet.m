classdef neuralNet < handle
    % define a neural net
    properties
        widths % vector
        depth % integer
        weightMatrices % cell array
        layers % cell array
        trainingError % object
        sgd % object
    end
    
        %% class dependencies
%                        - - activation - - 
%                       |                  |
%                       |                  |
%     - - neuralNet - - neuralNetLayer - - neuron
%    |    |
%    |    |
%   sgd   trainingError - - lossModel
%    |    |
%    |    |
%     - - probabilityLaw
        %%
    methods
        function obj = neuralNet(widths)
            % widths = [inputDimension #neuronsInLayer1 ... outputDimension]
            obj.widths = widths;
            obj.depth = length(widths)-1;
            obj.weightMatrices = cell(1,obj.depth);
            for layerNumber = 1:obj.depth
                % include bias term
                obj.weightMatrices{layerNumber} = randn(widths(layerNumber)+1,widths(layerNumber+1));
            end

            obj.layers = cell(1,obj.depth);
            for layerNumber = 1:obj.depth
                obj.layers{layerNumber} = neuralNetLayer(obj,layerNumber);
            end
            
            obj.trainingError = trainingError(obj);
            obj.sgd = sgd(obj);
        end
            
        function [inputs, netInputs, hiddenOutputs, outputs] = feedforward(obj,inputs)
            % size(input) = [inputDimension numberOfSamples]
            % size(output) = [outputDimension numberOfSamples]
            netInputs = cell(1,obj.depth);
            hiddenOutputs = cell(1,obj.depth);
            HO = inputs;
            for layerNumber = 1:obj.depth
                [NI, HO] = obj.layers{layerNumber}.feedforward(HO);
                netInputs{layerNumber} = NI;
                hiddenOutputs{layerNumber} = HO;
            end
            
            outputs = hiddenOutputs{end};
        end
        
    end
        
end