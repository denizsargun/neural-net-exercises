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
        numberOfSamples = 100;
        classesRealized
        observation
        data
    end
    
    methods
        function obj = probabilityLaw()
            obj.classDistribution = 1/obj.numberOfClasses*ones(obj.numberOfClasses,1);
        end
        
        function realize_classes(obj)
            obj.classesRealized = randi(obj.numberOfClasses,obj.numberOfSamples,1)-1;
        end
        
        function observe(obj)
            obj.realize_classes()
            obj.observation = sqrt(obj.variance)*randn(obj.numberOfSamples,1)+obj.classesRealized;
            obj.data = [obj.classesRealized, obj.observation];
        end
        
    end
        
end