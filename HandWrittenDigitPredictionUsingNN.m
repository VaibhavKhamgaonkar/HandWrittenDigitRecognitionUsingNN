%###############################################Handwritten digit recognation task #####################################
%Architecture contains 3 layes (1 - i/p, 1 - hidden, 1 - o/p), Hidden layer has 25 units and o/p layer has 10 units which will predict 0 to 9 digit (0 being 10)
%The image which has dimension of 20X20 (i.e. 400 pixels ) is applied at this layer
%As per the architectire Theta1 and Theta2 will have the dimensions as follows 
%Theta1 = (next layer units) x (current layer units + 1)  ==> 25 * (400 + 1) ==> 25*401
%Theta2 = (next layer units) x (current layer units + 1)  ==> 10 * (25 + 1) ==> 10*26

								%Steps to compute parameter
%Prerequiste : randomly initialise parameters for Theta1 and Theta2 
epsilon = 0.12;
Theta1 = rand(25,401) * (2 * epsilon) - epsilon
Theta2 = rand(10,26) * (2 * epsilon) - epsilon
inputLayerSize = 400;
outputLayerSize = 10;
hiddenLayerSize = 25;
lambda = 0.0;
%unroll the Theta parameters for Cost computaion and further operations
nn_Theta = [Theta1(:); Theta2(:)];

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

 
 % create a **Sigmoid function*** t ocalculate the hypothesis 
 function [ h ] = sigmoid(z)
	h = 1/ (1 + exp(-z))
 end
 
 
%1. Apply Forward Proogation and compute hTheta(x) or a3 
 %Creating a **Cost function (Unregularised  + Regularised)***
 function [J grad] = NNCostFunction(X,y,ThetaUnrolled,lambda,m,inputLayerSize,outputLayerSize,hiddenLayerSize)
	
	k = outputLayerSize;
 % first thind roll the Theta back to their original shapes.
	Theta1 = reshape(ThetaUnrolled (1:hiddenLayerSize*(inputLayerSize+1)),hiddenLayerSize, (inputLayerSize+1));
	Theta2 = reshape(ThetaUnrolled((size(Theta1,1)*size(Theta1,2))+1:end), k, (hiddenLayerSize+1));
	
	
	% compute the o/p layer activation by performing forward Propagation and compute cost
	% First set up Bias unit for input and hidden layer
	a1 = [ones(m,1),X];
	a2 = [ones(m,1), sigmoid(X*Theta1')];  %----> X = 5000 * 401 ; Theta1 = 25 * 401 so X * Theta' = 5000x401 x 401 x 25 ==> 5k x 25
	% and a2 wil have 5K X 26 becasue of the bias unit.
	a3 = sigmoid (a2 * Theta2'); %---> a2 = 5K x 26 and Theta2' = 26*10 ==> a2*Theta2 = 5K * 10
	
	%now rearranging y to 1 * 10 vector as there are 10 o/p units for each training example in NN but y has only 1 so reaaranging y to 10 vector length
	yNew = repmat([1:k], m, 1)	== repmat(y, 1, k); % --. the 1st part will create a matrix of length m (rows) and k rows with values 1 to k in each element of row and 1 is for one column. 2nd part will apply the same logic creating a matrix of y with 1 row and k column with values of y in each each row.
	
	cost =  sum(yVec .* log(a3) + (1-yVec) .* log(1-a3));
	J = (-1/m) * sum(cost); %---> unregularised cost 
	
	% Now compute the regaularsied cost.
	Theta1WithoutBias = Theat1(:,2:end);
	Theta2WithoutBias = Theat2(:,2:end); %--> excluding the bias
	
	reg_term = (lambda/(2*m)) * ( sum(Theta1WithoutBias .^2) + sum(Theta2WithoutBias .^2) )
	
	% final Cost with regularisation
	J = J + reg_term;
	
	% going for Back Propoation for parameter calculation
	del(L)
	
	 
 end

%calculating the Sigmoind gradient(i.e partial derivative of Sigmoind function

function [g] = sigmoidGradient(z)
	h = sigmoid(z);
	g = h .* (1 - h);
end 


% Inistialising some parameters 




