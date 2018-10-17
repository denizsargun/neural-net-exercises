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
            obj.lossModel = lossModel();
        end
        
        function error = on_data(obj,data)
            % data = [correct label; features]
            correctLabels = data(1,:);
            features = data(2:end,:);
            [~, ~, ~, classEstimates] = obj.neuralNet.feedforward(features);
            error = obj.lossModel.loss(correctLabels,classEstimates);
        end
        
        function error = random(obj,sampleSize)
            % data = [correct label; features]
            data = obj.probabilityLaw.sample(sampleSize);
            correctLabels = data(1,:);
            features = data(2:end,:);
            [~, ~, ~, classEstimates] = obj.neuralNet.feedforward(features);
            error = obj.lossModel.loss(correctLabels,classEstimates);
        end
        
    end
    
end