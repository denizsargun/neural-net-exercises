classdef probabilityLaw < handle
    % define a probability law that describes the observations
    % let Y={0,1,...,m} and p_Y(y)=1/m
    % let X=R and p_{X|Y}(x|y)=N(c*y,var)(x)
    properties
        numberOfClasses = 2;
        classDistributionType = 'uniform';
        classDistribution
        c = 1;
        variance = 10;
    end
    
    methods
        function obj = probabilityLaw()
            obj.classDistribution = 1/obj.numberOfClasses*ones(obj.numberOfClasses,1);
        end
        
        function realize_classes(obj,numberOfSamples)
            obj.classesRealized = randi(obj.numberOfClasses,1,numberOfSamples)-1;
        end
        
        function output = sample(obj,numberOfSamples)
            classesRealized = obj.realize_classes(numberOfSamples);
            features = sqrt(obj.variance)*randn(1,numberOfSamples)+classesRealized;
            output = [classesRealized; features];
        end
        
    end
        
end