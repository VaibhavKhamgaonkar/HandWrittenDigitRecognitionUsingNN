# MicrochipTestPrediction

The program predict whether A microchip is passed the qa test bested on two different test result. Its a logistic/classification problem.

Copy the data files "*.txt" to your local drive. Open the Matlab file and copy the code from it. Paste it in octave or Matlab and run it. Make sure your working directory contains data files that you copied from here.

# NOTE: 
1. Apha and lambda can change the decision boundary and thus can affect the accuricy. Here Alpha and lambda are set to best suited values i.e 0.01 which gives accuricy of 89.5857% on Training data.

2. When testing any new test data having two features (as used to train the model), user have to use ScaleFeature functionality to raise the features to 6th order polynomial equation so as to work properly.
i.e. user have to pass the testdata to ScaleFeature function which will convert the test data into higher order equation.

