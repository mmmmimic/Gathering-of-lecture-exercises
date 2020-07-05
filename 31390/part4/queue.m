classdef queue
    % Data Structure STACK
    % LIFO: Last In, First Out
    properties
        data = [];
    end
    
    methods
        function obj = queue()
            %STACK Construct an instance of this class
            %   Detailed explanation goes here
            obj.data = [];
        end
        
        function obj = push(obj,arg)
            obj.data = [obj.data, arg];
        end
        
        function [d, obj] = pop(obj)
           d = obj.data(1);
           %obj.data = obj.data(2:end);
           obj.data(obj.data==d) = [];
        end
            
    end
end

