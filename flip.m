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
test_size = 100;
flip_number = 3;
correct_times = zeros(flip_number, size(input_matrix,2) - 1);
correct_rate = zeros(flip_number, size(input_matrix,2) - 1);
for digit_num = 2 : size(input_matrix,2)
    layer = layer.setInputMatrix(input_matrix(:,1 : digit_num));
    layer = layer.setOutputMatrix(output_matrix(:,1 : digit_num));
    %layer = layer.hebbRule();
    layer = layer.pseudoInverse();
    for j = 1 : flip_number  % 2 * j is the number of fliped bytes
        for i = 1 : test_size
            for k = 1 : digit_num
                noiseInput = addNoise(input_matrix(:,k),2 * j);
                prediction = round(layer.forward(noiseInput));
                if isequal(output_matrix(:,k),prediction)
                    correct_times(j,digit_num - 1) = correct_times(j,digit_num - 1) + 1;
                end
            end
        end
        disp("number of fliped bits " + 2 * j);
        correct_rate(j,digit_num - 1) = correct_times(j,digit_num - 1) / (test_size * digit_num);
        disp("correct rate after storing " + k + " digits with " + j + " digits fliped is");
        disp(correct_rate(j,digit_num-1));
    end
end

% plot
flip_num = [2,4,6];
digitNum = [2,3,4,5,6];
for i = 1 : size(flip_num,2)
    plot(digitNum, correct_rate(i,:), "o-","DisplayName",i + " digits");
    hold on;
end
legend("show");
hold off;
