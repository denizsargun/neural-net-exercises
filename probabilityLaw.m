classdef probabilityLaw < handle
    % define a probability law that describes the observations
    % let Y={0,1,...,m} and p_Y(y)=1/m
    % let X=R and p_{X|Y}(x|y)=N(c*y,var)(x)
    properties
        numberOfLabels = 2;
        labelDistributionType = 'uniform';
        labelDistribution
        c = 1;
        variance = 1;
    end
    
    methods
        function obj = probabilityLaw()
            obj.labelDistribution = 1/obj.numberOfLabels*ones(obj.numberOfLabels,1);
        end
        
        function output = realize_labels(obj,numberOfSamples)
            output = randi(obj.numberOfLabels,1,numberOfSamples)-1;
        end
        
        function output = sample(obj,numberOfSamples)
            labelsRealized = obj.realize_labels(numberOfSamples);
            features = sqrt(obj.variance)*randn(1,numberOfSamples)+labelsRealized;
            output = [labelsRealized; features];
        end
        
    end
        
end