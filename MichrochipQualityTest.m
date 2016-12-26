% Predicting the failure chances of a microchip based on various test outcomes.

%Clearing the Data
clear ; close all; clc;

% Importing the training data 
data = load ('ex2data2.txt');
X = data(:,1:2);
y = data(:,end);

%plot(X(:,1),X(:,2), 'markersize', 7,'ko');

%creating a function that will going to plot the data based on the the result
function[] = plotting(X,y)
  pos = find( y == 1);
  neg = find(y ==0);
  
  plot(X(pos,1), X(pos,2), 'MarkerSize', 7,'k+','LineWidth', 2);
  hold on;
  plot(X(neg,1), X(neg,2), 'MarkerFaceColor', 'y','ko');
  legend('Passed Quality Test', 'Failed');
  xlabel('QA test 1');
  ylabel('QA test 2');
  %hold off;
end

plotting(X,y);

%as this ais not a stright classification problem we have to scale ... 
%the Features to higher order so as to fit the graph and create a non linear boundry   

  function [temp] = ScaleFeatures(X)
      %seperating the features
      X1 = X(:,1);
      X2 = X(:,2);
      %going for 6 oder polynomial function
      degree = 6;
      % this function wil going to create more features for the exisiting two features
      % X1,X2, X1*X2, X1,X2^2, X1^2*X2 ... and so on.
      temp = ones(size(X1(:,1)));
      for i = 1:degree
        for j = 0:i
          temp(:,end+1) = (X1.^(i-j)).* (X2.^(j));
          
        end
      end

  end
  
%hypothesis function
  function[h] = LogisticHyphothesis(z)

    h = 1./(1+exp(-z));

  end


%size(X)
%scaling the Featues to 6 order polynomial
X = ScaleFeatures(X); % this has already added column of ones so no need to added extra x0 column

%defining initial parameter
theta = zeros(size(X,2),1); % defining theta for 28 features in onesingle columns
lambda = 0.01;
alpha = 0.01;
m = length(y);

%Creating Cost function (regularised)
  function [J grad] = CalculateCost(X,y,theta,alpha,lambda,m)
    h = LogisticHyphothesis(X*theta);
  %regularisation parameter calcuation
  % for computing regularisied cost
    regCost = (lambda/(2*m)) * sum(theta.^2);  
  % for computing regularised grad
    regGrad = (lambda/m) * [0; theta(2:end,:)]; 
    
    %cost
    J = (-1/m)*(y' * log(h) + (1-y)' * log(1-h)) + regCost;
    %grad
    grad = (alpha/m)*(X' * (h-y)) + alpha*regGrad;
  end 

%calculating Gradient and cost using advanced optimisation 
options = optimset('GradObj','on', 'MaxIter',400);
[theta cost exitFlag] = fminunc(@(t)CalculateCost(X,y,t,alpha, lambda,m), theta, options);


%Plotting the decision boundry
  u = linspace(-1, 1.5, 50);
  v = linspace(-1, 1.5, 50);

  z = zeros(length(u), length(v));
  % Evaluate z = theta*x over the grid
  for i = 1:length(u)
      for j = 1:length(v)
          z(i,j) = ScaleFeatures([u(i), v(j)])*theta; % as ScaleFeatures taskes only one Argument i.e matrix thus u(i) and v(i) as club together.
      end
  end
  z = z'; % important to transpose z before calling contour

  % Plot z = 0
  % Notice you need to specify the range [0, 0]
  contour(u, v, z, [0, 0], 'LineWidth', 2)
  legend('Passed Quality Test', 'Failed', "Decision Boundary");
  title(sprintf('Prediction when lambda = %g, alpha = %g', lambda,alpha));
  hold off      

  
% prediction Accuricy on given data set.

accuricy = LogisticHyphothesis(X*theta);

fprintf("Accuricy of the prediction is about \n:%f\n", mean(accuricy(y = 1)) * 100);  
 
