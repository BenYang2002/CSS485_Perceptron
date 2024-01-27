classdef PerceptronLayer
    properties
        weight_matrix
        bias

        %name of the activation function
        transfer
        
        % the whole input set
        input_matrix

        %the current single input
        input_vector

        %the whole standard/correct output
        output_matrix
        
        %the current standard/correct output
        output_vector
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

        % input: takes one parameter
        % input: the input matrix
        % it serves as the setter for input matrix
        function obj = input_setter(obj,input)
            % this assumes each column is a set of input
            if size(obj.weight_matrix,2) ~= size(input,1)
                error("Perceptron.input: dimention of the training set" + ...
                    " does not match with weight matrix");
                return;
            end
            obj.input_matrix = input;
        end

        % output: takes one parameter
        % output: the output matrix
        % it serves as the setter for output matrix
        function obj = output_setter(obj,output)
            % this assumes each column is a set of input
            if size(obj.weight_matrix,1) ~= size(output,1)
                error("Perceptron.input: dimention of the output" + ...
                    " does not match with weight matrix");
                return;
            end
            obj.output_matrix = output;
        end

        % errorLoss takes in two parameters:
        % a: the prediction
        % t: the expected/correct output
        % return the vector of t - a
        function output = errorLoss(obj,a,t)
            if size(a,2) ~= 1 || size(t,2) ~= 1
                erro("perceptron.errorLoss: expect a vector");
                return;
            elseif size(a,1) ~= size(t,1)
                disp("perceptron.errorLoss:  dimension of a and t does not" + ...
                    " match ");
                return;
            end
            output = t - a;
        end

        % backward takes in one parameters:
        % error: the vector of errors
        % this function update the bias and weight matrix
        function obj = backward(obj,error)
            % this assumes error is a vector
            % if the dimension does not match with neuron number, display
            % error msg
            if size(error,1) ~= size(obj.weight_matrix,1)
                error("perceptron.backward: dimension of error does not" + ...
                    " match with neuron number");
                return;
            end

            % update bias
            obj.bias = obj.bias + error;

            % update weight matrix
            fix_matrix = (obj.input_vector * error')';
            obj.weight_matrix = obj.weight_matrix + fix_matrix;
        end
        
        % print does not take any parameter
        % this function prints out the weight matrix and bias
        function print(obj)
            disp("weight_matrix: ");
            disp(obj.weight_matrix);
            disp("bias: ");
            disp(obj.bias);
        end

        function obj = learn2(obj);
            correct = false ;
            count = 1;
            while (~correct & count <= 1000)
                correct = true;
                for i = 1:size(obj.input_matrix,2)
                    obj.input_vector = obj.input_matrix(:,i);
                    obj.output_vector = obj.output_matrix(:,i);
                    prediction = obj.forward1(obj.input_vector);
                    error = obj.errorLoss(prediction,obj.output_vector);
                    if ~all(error == 0)
                        correct = false;
                    end
                    obj = obj.backward(error);
                    count = count + 1;
                end
            end
            if correct
                obj.print();
            end
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
        % forward can also take a matrix input
        function output = forward1(obj,input)
            netinput = obj.weight_matrix * input + obj.bias;
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
