%Create training set
p1 = [1, 4];
t1 = 0;

p2 = [1, 5];
t2 = 0;

p3 = [2, 4];
t3 = 0;

p4 = [2, 5];
t4 = 0;

p5 = [3, 1];
t5 = 1;

p6 = [3, 2];
t6 = 1;

p7 = [4, 1];
t7 = 1;

p8 = [4, 2];
t8 = 1;

patterns = [p1; p2; p3; p4; p5; p6; p7; p8];
targets = [t1; t2; t3; t4; t5; t6; t7; t8];

%Initialize a PerceptronLayer neural network 
perceptron = PerceptronLayer(2, 1, 0.1);
trainPerceptron(perceptron, patterns, targets)


%this function initializes and trains a PerceptronLayer neural network to classify two-dimensional vectors according to the training set provided
function trainPerceptron(perceptron, patterns, targets)
    %initialize the neural network
    arguments
        perceptron PerceptronLayer
        patterns double
        targets double
    end

    %train the neural network
    perceptron.learn(patterns, targets);
end

