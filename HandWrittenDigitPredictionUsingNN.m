%###############################################Handwritten digit recognation task #####################################
%Architecture contains 3 layers (1 - i/p, 1 - hidden, 1 - o/p), Hidden layer has 25 units and o/p layer has 10 units which will predict 0 to 9 digit (0 being 10)
%The image which has dimension of 20X20 (i.e. 400 pixels ) is applied at this layer
%As per the architecture Theta1 and Theta2 will have the dimensions as follows 
%Theta1 = (next layer units) x (current layer units + 1)  ==> 25 * (400 + 1) ==> 25*401
%Theta2 = (next layer units) x (current layer units + 1)  ==> 10 * (25 + 1) ==> 10*26


clear ; close all; clc


								%Steps to compute parameter
%Prerequisite : randomly initialise parameters for Theta1 and Theta2 
epsilon = 0.12;
inputLayerSize = 400;
outputLayerSize = 10;
hiddenLayerSize = 25;
Theta1 = rand(hiddenLayerSize,(inputLayerSize+1)) * (2 * epsilon) - epsilon;
Theta2 = rand(outputLayerSize, (hiddenLayerSize+1)) * (2 * epsilon) - epsilon;
lambda = 1;

%unroll the Theta parameters for Cost computation and further operations
ThetaUnrolled = [Theta1(:); Theta2(:)];



%loading the  data set
fprintf("load the dataset\n");
load('ex4data1.mat');

%loading random image for visualization
	sel = randperm(length(X));
	for i = 1: length(X)
		temp = reshape(X(sel(i),:),20,20);
		imshow(temp);
		pause;
		fprintf("(ctr + c) 2 times followed by 'Esc' to abort\n");
	end
%Calculate the size of training Examples 
 m = size(X,1);

 
 % create a **Sigmoid function*** to calculate the hypothesis 
 function [ h ] = sigmoid(z)
	h = 1./(1 + exp(-z));
 end
 
 %calculating the Sigmoid gradient(i.e partial derivative of Sigmoid function
function [g] = sigmoidGradient(z)
	h = sigmoid(z);
	g = h .* (1 - h);
end 

 
%1. Apply Forward Prorogation and compute hTheta(x) or a3 
 %Creating a **Cost function (Unregularised  + Regularised)***
 function [J grad] = NeuralNetCostFunction(X,y,ThetaUnrolled,lambda,m,inputLayerSize,outputLayerSize,hiddenLayerSize)
		k = outputLayerSize;
	 % first thing roll the Theta back to their original shapes.
		Theta1 = reshape(ThetaUnrolled (1:hiddenLayerSize*(inputLayerSize+1)),hiddenLayerSize, (inputLayerSize+1));
		Theta2 = reshape(ThetaUnrolled((size(Theta1,1)*size(Theta1,2))+1:end), k, (hiddenLayerSize+1));
			
		% compute the O/p layer activation by performing forward Propagation and compute cost
		% First set up Bias unit for input and hidden layer
		a1 = [ones(m,1),X];
		a2 = [ones(m,1), sigmoid(a1*Theta1')];  %----> X = 5000 * 401 ; Theta1 = 25 * 401 so X * Theta1' = 5000x401 x 401 x 25 ==> 5k x 25
		% and a2 will have 5K X 26 because of the bias unit.
		a3 = sigmoid (a2 * Theta2'); %---> a2 = 5K x 26 and Theta2' = 26*10 ==> a2*Theta2 = 5K * 10
		
		%now rearranging y to 1 * 10 vector as there are 10 o/p units for each training example in NN but y has only 1 so rearranging y to 10 vector length
		yVec = repmat([1:k], m, 1)== repmat(y, 1, k); % --. the 1st part will create a matrix of length m (rows) and k rows with values 1 to k in each element of row and 1 is for one column. 2nd part will apply the same logic creating a matrix of y with 1 row and k column with values of y in each each row.
		
		cost =  sum(yVec .* log(a3) + (1-yVec) .* log(1-a3));
		J = (-1/m) * sum(cost); %---> un-regularised cost 
		
		% Now compute the regularised cost.
		Theta1WithoutBias = Theta1(:,2:end);
		Theta2WithoutBias = Theta2(:,2:end); %--> excluding the bias
		
		reg_term = (lambda/(2*m)) * ( sum(sum(Theta1WithoutBias.^2)) + sum(sum(Theta2WithoutBias.^2)) );
		
		% final Cost with regularisation
		J = J + reg_term;
		
		%setting up delta parameters
		delta1 = zeros(size(Theta1));
		delta2 = zeros(size(Theta2));
		
		% going for Back Propagation for parameter calculation
		for i = 1 : m
			a1i = [a1(i,:)']; %---> 401 x 1 matrix
			z2i = Theta1 * a1i;  % ---->25 x 1 matrix
			a2i = a2(i,:); %--> 26 x 1
			h1i = a3(i,:)';   % --> 10 x 1 matrix
			yVeci = yVec(i,:)';  %---> 10 x 1 matrix
			
			% compute del values from o/p layer
			d3i = h1i - yVeci;
      d2i = ((Theta2)' * d3i ) .* sigmoidGradient([1;z2i]);
  
      delta1 = delta1 + (d2i(2:end,:)*a1i');
      delta2 = delta2 + ( d3i * a2i );
      
      %d3i = h1i - yVeci; %--> 10 x 1
			%d2i = (Theta2' * d3i) .* sigmoidGradient([1;z2i]);  % ---> need to consider bias as well
			%delta1 = delta1 + d2i(2:end,:) * a1i' ; %--. while calculating Delta1 bias term should be excluded.
			%delta2 = delta2 + (d3i * a2i') ;
		end
	%now computing the Regularised Gradient theta
	Theta1ZeroBias = [zeros(size(Theta1,1),1), Theta1WithoutBias];
	Theta2ZeroBias = [zeros(size(Theta2,1),1), Theta2WithoutBias];

	Theta1_grad = (1/m)*(delta1);
	Theta2_grad = (1/m)*(delta2);

	Theta1_grad = Theta1_grad + (lambda/m) * Theta1ZeroBias;
	Theta2_grad = Theta2_grad + (lambda/m) * Theta2ZeroBias;

	% return the Grad by unrolling it
	grad = [ Theta1_grad(:) ; Theta2_grad(:)];
	
 end

 % Function to predict the digit
function [pred] = Prediction (Theta1,Theta2, X)
	m = size(X,1);
	outputLayerSize = size(Theat2,1);

	h1 = sigmoid( [ones(m,1), X] * Theat1');
	h2 = sigmoid([ones(m,1), h1 ] * Theta2');

	[max_value pred ] = max(h2, [], 2);

end

%calculate the cost and Gradient using advanced optimisation
% defining the parameter for cost function
 costcompute = @(t) NeuralNetCostFunction(X,y, t, lambda, m, inputLayerSize,outputLayerSize,hiddenLayerSize);
 
options = optimset('MaxIter', 50);

% given below two advanced optimisatin function, use any one of them. 
% 1. Using Fminunc
[ThetaUnrolled J] = fminunc(costcompute, ThetaUnrolled, options);

%2. using Fmincg
[ThetaUnrolled J] = fmincg(costcompute, ThetaUnrolled, options);


%Reshape thet1 and theta2 from unrolled version 
Theta1 = reshape(ThetaUnrolled (1:hiddenLayerSize*(inputLayerSize+1)),hiddenLayerSize, (inputLayerSize+1));
Theta2 = reshape(ThetaUnrolled((size(Theta1,1)*size(Theta1,2))+1:end), outputLayerSize, (hiddenLayerSize+1));


% finding the Accuracy in Training set

pred = Prediction(Theat1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100); 









