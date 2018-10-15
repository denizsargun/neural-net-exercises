classdef activation < handle
    % activation nonlinearity
    properties
        type = 'sigmoid';
        parameter = [1; 0]; % sigmoid parameter
    end
    
    methods
        %% input output options
        function output = output(obj,input)
            output = obj.sigmoid(input);
%             output = obj.recLin(input);
%             output = obj.softPlus(input);
        end
        
        function output = sigmoid(obj,input)
            output = sigmf(input,obj.parameter);
        end
        
        function output = recLin(obj,input)
            output = max(0,input);
        end
        
        function output = softPlus(obj,input)
            output = log(1+exp(input));
        end
        
        %% derivative options
        function output = diff(obj,input)
            output = obj.diffSigmoid(input);
%             output = obj.diffRecLin(input);
%             output = obj.diffSoftPlus(input);
        end
        
        function output = diffSigmoid(obj,input)
            output = obj.parameter(1)*sigmf(input,obj.parameter)*(1-sigmf(input,obj.parameter));
        end
        
        function output = diffRecLin(obj,input)
            output = 1/2*(sign(input)+1);
        end
        
        function output = diffSoftPlus(obj,input)
            output = sigmf(input,[1; 0]);
        end
        
    end
    
end