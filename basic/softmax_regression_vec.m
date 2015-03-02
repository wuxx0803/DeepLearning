function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  %%% YOUR CODE HERE %%%
  
  % only use num_classes-1 columns, since the last column is always assumed 0
  % exclude the effect of y(i) == 10
  % tX = [theta,zeros(n,1)]'*X;
  tX = theta'*X;
  rot_tX = tX';
  
  I = sub2ind(size(rot_tX),1:size(rot_tX,1),y);
  f = sum(rot_tX(I));
  
  % exlast_tX = exp(tX(1:end-1,:));
  exlast_tX = exp(tX);
  denominator = sum(exlast_tX,1);
  
  f = - f + sum(log(denominator));
  
%   for k = 1:num_classes-1
%       temp = zeros(1,m);
%       temp(y==k) = 1;
%       
%       g(:,k) = - X*(temp - bsxfun(@rdivide,exp(theta(:,k)'*X), denominator))';
%   end
  g = X * bsxfun(@rdivide, exlast_tX,denominator)';
  
  xy = zeros(m,num_classes);
  idx = sub2ind(size(xy),1:size(xy,1),y);
  xy(idx) = 1;
  xy = xy(:,1:end-1);
  
  g = g - X*xy;
  
  g=g(:); % make gradient a vector for minFunc

