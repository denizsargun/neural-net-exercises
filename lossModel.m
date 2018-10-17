classdef lossModel < handle
    properties
        lossType = 'averageSquared';
    end
    
    methods
        function loss = loss(obj,groundTruth,estimation)
            loss = obj.average_squared_loss(estimation,groundTruth);
%             loss = obj.total_squared_loss(estimation,groundTruth);
%             loss = obj.max_loss(estimation,groundTruth);
        end
        
    end
    
    methods (Static)
        function loss = average_squared_loss(estimation,groundTruth)
            numberOfSamples = length(estimation);
            loss = 1/numberOfSamples*sum((estimation-groundTruth).^2);
        end
        
        function loss = total_squared_loss(estimation,groundTruth)
            loss = sum((estimation-groundTruth).^2);
        end
        
        function loss = max_loss(estimation,groundTruth)
            loss = max(abs(estimation-groundTruth));
        end
        
    end
    
end