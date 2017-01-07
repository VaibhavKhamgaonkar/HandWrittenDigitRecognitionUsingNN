function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Alternate way of getting Theta2
% temp = reshape(nn_params(size(Theta1,1)*size(Theta1,2) +1 : end), ...
% num_labels, (hidden_layer_size + 1));             
                 
                 
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


k = num_labels;
% implementing Forward  Propagation method
%Adding Biasing to The Input layer
%X = [ones(size(X,1),1),X];
a1 = [ones(m, 1) X];
a2 = [ones(m ,1) sigmoid(a1 * Theta1')];
h = sigmoid(a2*Theta2');


% As the o/p layer has 10 unit and y has only one unit ...
%so it has to convert from 1 unit to 10 so as to ...
%perfomr the J calculation correctly.
 
 yVec = repmat([1:k],m,1) == repmat(y, 1, k);
  
J =  (-1/m)* sum(sum((yVec .* log(h) + (1-yVec) .* log(1-h))));

%a1 = [ones(m, 1) X];
%a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
%h = sigmoid(a2 * Theta2');

% Constructing a vector of result ex: for 5 of 10 the 1 should be at 
% fifth position [0 0 0 0 1 0 0 0 0 0] where rows are training set samples
%yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

%cost = -yVec .* log(a3) - (1 - yVec) .* log(1 - a3);

%J = (1 / m) * sum(sum(cost));

%part2 : regularised Cost

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

J = J + (lambda/(2*m))*(sum(sum(Theta1NoBias.^2)) + sum(sum(Theta2NoBias.^2)));

%part 3 : back Propagation
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));


  for i = 1 : m
    h1i = h(i,:)';
    a1i = a1(i,:)';
    a2i = a2(i,:)';
    yVeci = yVec(i,:)';
    %computed d3i for each element
d3i = h1i - yVeci;
    %compute Delta 2 for each element
    z2i = [1; Theta1 * a1i];
d2i = (Theta2' * d3i) .* sigmoidGradient(z2i);
    
    delta1 = delta1 + (d2i(2:end) * a1i'); 
    delta2 = delta2 + (d3i * a2i');
    
    end
    
    Theta_grad = (1/m) * delta1;
    Theta2_grad = (1/m) * delta2;
 

    
Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;
    

%Reguarised Theta_grad
  Theta1zeroBias  = [zeros(size(Theta1,1),1), Theta1NoBias];
  Theta2zeroBias  = [zeros(size(Theta2,1),1), Theta2NoBias];
  Theta1_grad =  Theta1_grad = (1 / m) * delta1 + (lambda/m) * Theta1zeroBias ; 
  Theta2_grad =  Theta2_grad = (1 / m) * delta2 + (lambda/m) * Theta2zeroBias ; 
  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
