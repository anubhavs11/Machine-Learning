function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

[i,j] = size(X);
j = j+1;
a1 = zeros(i,j);
a1(:,1) = [1];
a1(:,2:j) = X;

z2 = a1*Theta1';
z2 = sigmoid(z2);

[i,j] = size(z2);
j = j+1;
a2 = zeros(i,j);
a2(:,1) = [1];
a2(:,2:j) = z2;

a3 = a2*Theta2';
[i,p] = max(a3,[],2);
%[predict max,index max]
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
% 
% =========================================================================


end
