classdef Queue<handle
    properties(SetAccess = private)
        matrix
    end
    methods
        function push(Queue, dat)
            if size(dat,1)~=2
                disp('Push Error: Dat Size Error')
            else
                Queue.matrix = [Queue.matrix, dat]; 
            end
        end
    
        function dat = pop(Queue)
            dat = Queue.matrix(:,1);
            Queue.matrix = Queue.matrix(:,2:end);
        end
    end
    methods(Static)
        
    end
    events
        
    end
end