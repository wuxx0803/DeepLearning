%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
cost = 0;
Wgrad = zeros(size(W));

% Forward Propagation RICA layer
h = W*x;
r = W'* h;

% Sparsity Cost
K = sqrt(h.^2 + params.epsilon);
sparsity_cost = params.lambda * sum(K(:));
K = 1./K; % for easily calculation at final Wgrad stage

% Reconstruction Loss
diff = r - x;
reconstruction_cost = 0.5* sum(sum(diff.^2));

cost = sparsity_cost + reconstruction_cost;

% Back Propagation
% Output Layer
output_derv = diff;
W2grad = output_derv * h';

% Hidden Layer
hidden_derv = W * output_derv;
hidden_derv = hidden_derv + params.lambda * (h.*K);
W1grad = hidden_derv * x';

Wgrad = W1grad + W2grad';

% straight forward but contains duplicate computations
% cost = params.lambda.* sum(sum(sqrt((W*x).^2 + params.epsilon))) + ...
%         sum(sum((W'*W*x - x).^2))/2;
%     
% Wgrad = W*(W'*W*x-x)*x'+ W*x*(W'*W*x-x)'...
%     + params.lambda.*W*x./sqrt((W*x).^2 + params.epsilon)*x';
%cost = cost./params.m;
%Wgrad = Wgrad./params.m;
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);