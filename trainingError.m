classdef trainingError < handle
    properties
        neuralNet
        probabilityLaw
        lossModel
    end
    
    methods
        function obj = trainingError(neuralNet)
            obj.neuralNet = neuralNet;
            obj.probabilityLaw = probabilityLaw();
        end
        
        function error = onData(obj,data)
            % data = [correct label; features]
            correctLabels = data(1,:);
            features = data(2:end,:);
            [~ classEstimates] = obj.neuralNet.feedforward(features);
            error = obj.lossModel.loss(correctLabels,classEstimates);
        end
        
        function error = random(obj,sampleSize)
            % data = [correct label; features]
            data = 
            correctLabel = data(1,:);
            input = data(2:end,:);
            classEstimate = obj.feedforward(input);
            output = obj.lossFunction(classEstimate-correctLabel);
        end
        
    end
    
    methods (Static)
        function output = lossFunction(difference)
            % L2 loss
            output = norm(difference);
        end
        
    end
    
end