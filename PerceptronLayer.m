classdef PerceptronLayer
    properties
        weight_matrix
        bias
        transfer
    end
    % this is the constructor where input 1 is the weight,
    % input 2 is the bias and the third argument is the activation function name
    methods
        function obj = PerceptronLayer(input1,input2,funcName)
            div1 = size(input1);
            div2 = size(input2);
            if div1(1) == 1 & div1(2) == 1 & div2(1) == 1 & div2(2) == 1
                obj.weight_matrix = 2 * rand(input2,input1) - 1;
                obj.bias = 2 * rand(input2,1);
            else 
                obj.weight_matrix = input1;
                % this makes sure bias is a vector not a row
                if size(input2,1) == 1
                    obj.bias = input2';
                else
                    obj.bias = input2;
                end
            end
            obj.transfer = funcName;
        end

        function output = errorLoss(obj,a,t)
            if size(a,2) ~= 1 || size(t,2) ~= 1
                disp("err:expect a vector");
                return;
            elseif size(a,1) ~= size(t,1)
                disp("err: dimension of a and t does not match ");
                return;
            end
            output = t - a;
        end

% this function let the perceptron learn about the weights and set the
% highest iteration to 1000
function output = learn(obj,input_set,output_set)
            %each row of the training set is a single set
            %each column of the output set is the correct output for each
            %set of input
            x = [obj.weight_matrix, obj.bias];
            %now z is the input set with 1 vector at the last column
            z = [input_set,ones(size(input_set,1))];
            z = z'; % each column of z is a input vector
            % a row of x times a column of z: ix * zi = the netinput for ith neuron
            error = 0;
            condition = false ;
            count = 1;
            while (~condition & count <= 1000)
                condition = true;
                for i = 1:size(input_set,1)
                    prediction = obj.forward1(input_set(i,:));
                    error = output_set(:,i) - prediction;
                    fix_matrix = [];
                    for j = 1:length(error)
                        fix_matrix = [fix_matrix ; input_set(i,:) * error(j) ];
                    end
                    obj.weight_matrix = obj.weight_matrix + fix_matrix;
                    obj.bias = obj.bias + error;
                    disp("count: " + count);
                    disp("bias: ");
                    disp(obj.bias);
                    disp("weight_matrix: ");
                    disp(obj.weight_matrix);
                    disp("error: ");
                    disp(error);
                    if ~all(error == 0)
                        condition = false;
                    end
                end
                count = count + 1;
            end
        end
% forward using matrix * vector
        function output = forward1(obj,input)
            netinput = obj.weight_matrix * input' + obj.bias;
            if strcmp(obj.transfer, "hardlim")
                output = hardlim(netinput);
            else
                output = hardlims(netinput);
            end
        end

 %forward through each neuron(iterate through each neuron)
        function outputVec = forward2(obj,input)
            outputVec = [];
            inVec = input';
            biasVec = obj.bias';
            for i = 1:size(obj.weight_matrix,1)
                ithNeuron = 0;
                ithNeuron = obj.weight_matrix(i,:) * inVec + biasVec(i);
                output = 0;
                if strcmp(obj.transfer, "hardlim")
                    output = hardlim(ithNeuron);
                else
                    output = hardlims(ithNeuron);
                end
                outputVec = [outputVec,output];
            end
            outputVec = outputVec';
        end

    end
    methods (Access = private) % set private
        %hardlim transformation func
        function result = hardlim(netInput)
            if netInput < 0
                result = 0;
            else
                result = 1;
            end
        end
        %hardlims transformation func
        function result = hardlims(netInput)
            if netInput < 0
                result = -1;
            else
                result = 1;
            end
        end
    end
end
