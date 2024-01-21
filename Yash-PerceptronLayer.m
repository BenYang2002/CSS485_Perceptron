%PerceptronLayer
%   PerceptronLayer is a class that represents a perceptron layer
%   It has a weight matrix, bias vector, and transfer function
classdef PerceptronLayer
    properties (Access = 'private')
        W           %weight matrix
        b           %bias vector
        transfer    %transfer function
    end
    methods
        %Constructor 
        %three arguments. Last argument is the name of the transfer function
        function obj = PerceptronLayer(arg1, arg2, arg3)
            %check dimensionality of first two arguments
            if (isscalar(arg1) && isscalar(arg2) && isstring(arg3))
                %situation 1: two scalar args. 
                %Create random weight matrix and bias vector of size arg1 x arg2 with random values between -1 and 1
                obj.W = rand(arg1, arg2) * 2 - 1;
                obj.b = rand(arg1, 1) * 2 - 1;
                obj.transfer = arg3;
            elseif (ismatrix(arg1) && isvector(arg2) && isstring(arg3))
                %situation 2: one weight matrix and bias vector
                obj.W = arg1;
                obj.b = arg2;
                obj.transfer = arg3;
            else
                error('Invalid arguments');
            end
        end
        
        %hardlim
        %hard limit transfer function
        function y = hardlim(~, x)
            if (isscalar(x))
                y = x >= 0;
            elseif (isvector(x))
                y = zeros(size(x));
                for i = 1:size(x, 1)
                    y(i) = x(i) >= 0;
                end
            else
                error('Invalid input');
            end
        end

        %hardlims
        %symmetric hard limit transfer function
        %params: x -> input parameter
        function y = hardlims(obj, x)
            one = 1;
            if (isvector(x))
                one = ones(size(x));
            end
            y = 2 * hardlim(obj, x) - one;
        end

        %forward
        %takes an input vector and returns the output vector
        %calls the transfer function which is specified in the constructor
        %first implementation using a for loop
        function output = forward(obj, input)
            %check dimensionality of input vector
            output = zeros(size(obj.W, 1), 1);

            if (size(input, 1) ~= size(obj.W, 2))
                error('Input vector must have the same number of rows as the number of columns in the weight matrix');
            end

            %iterate through each row of the weight matrix
            for i = 1:size(obj.W, 1)
                %calculate the net input and return the output
                weight = obj.W(i, :);
                bias = obj.b(i);
                n = weight * input + bias;
                output(i) = obj.activate(n);
            end
        end

        %forward2
        %takes an input vector and returns the output vector
        %calls the transfer function which is specified in the constructor
        %second implementation using matrix multiplication
        function output = forward2(obj, input)
            %check dimensionality of input vector
            if (size(input, 1) ~= size(obj.W, 2))
                error('Input vector must have the same number of rows as the number of columns in the weight matrix');
            else
                %calculate the net input and return the output
                n = obj.W * input + obj.b;
                output = obj.activate(n);      
            end
        end

        %display weights and biases
        function displayMatrices(obj)
            disp("Weight matrix: ");
            disp(obj.W);
            disp("Bias vector: ");
            disp(obj.b);
        end

        %activate
        %calls the transfer function which is specified in the constructor
        function y = activate(obj, x)
            if (obj.transfer == "hardlim")
                y = hardlim(obj, x);
            elseif (obj.transfer == "hardlims")
                y = hardlims(obj, x);
            else
                error('Invalid transfer function');
            end
        end
    end
end

