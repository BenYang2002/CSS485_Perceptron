% input for 0
input0 = [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]';
% input for 1
input1 = [-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]';
% input for 2
input2 = [1,1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1]';
% input for 3
input3 = [-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1]';
% input for 5
input4 = [1,-1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]';
% input for 3
input5 = [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,1,1,0,0,0,0,0]';

%normalization
%input0 = input0/norm(input0);
%input1 = input1/norm(input1);
%input2 = input2/norm(input2);
%input3 = input3/norm(input3);
%input4 = input4/norm(input4);
%input5 = input5/norm(input5);

%output matrix 
output_matrix = [input0,input1,input2,input3,input4,input5];
input_matrix = [input0,input1,input2,input3,input4,input5];

% train
layer = SupervisedHebbianLayer(30,30,"linear");
for pattern_num = 2 : size(input_matrix,2)
    layer = layer.setInputMatrix(input_matrix(:,1 : pattern_num));
    layer = layer.setOutputMatrix(output_matrix(:,1 : pattern_num));
    %layer = layer.hebbRule();
    layer = layer.pseudoInverse();
    test_size = 100;
    correct_times = zeros(1, pattern_num);
    for j = 1 : 3  % 2 * j is the number of fliped bytes
        for i = 1 : test_size
            for k = 1:pattern_num
                noiseInput = addNoise(input_matrix(:,k),2 * j);
                prediction = round(layer.forward(noiseInput));
                if isequal(output_matrix(:,k),prediction)
                    correct_times(k) = correct_times(k) + 1;
                end
            end
        end
        disp("number of fliped bits " + 2 * j);
        for k = 1:pattern_num
            disp("correct rate for pattern " + k + " is");
            disp(correct_times(k));
        end
        correct_times = zeros(1, pattern_num);
    end
end
