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
%             output = obj.rec_linear(input);
%             output = obj.soft_plus(input);
        end
        
        function output = sigmoid(obj,input)
            output = sigmf(input,obj.parameter);
        end
        
        function output = rec_linear(obj,input)
            output = max(0,input);
        end
        
        function output = soft_plus(obj,input)
            output = log(1+exp(input));
        end
        
        %% derivative options
        function output = diff(obj,input)
            output = obj.diff_sigmoid(input);
%             output = obj.diff_rec_linear(input);
%             output = obj.diff_soft_plus(input);
        end
        
        function output = diff_sigmoid(obj,input)
            output = obj.parameter(1)*sigmf(input,obj.parameter).*(1-sigmf(input,obj.parameter));
        end
        
        function output = diff_rec_linear(obj,input)
            output = 1/2*(sign(input)+1);
        end
        
        function output = diff_soft_plus(obj,input)
            output = sigmf(input,[1; 0]);
        end
        
    end
    
end