function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

% activationType = 'relu'; % Elapsed time is 395.22254 seconds. Accuracy is 98.34, 98.62(mean-normalized data)
activationType = 'sigmoid'; % Elapsed time is 436.91066 seconds. Accuracy is 97.00

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
% WHY THIS IS WRONG ???
% % we do activation at convolution layer
% activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
% % but we dont use activation at pool layer
% activationsPooled = cnnPool(poolDim, activations);


Wc_rotated = zeros(size(Wc));
for filterNum = 1 : numFilters
    Wc_rotated(:, :, filterNum) = rot90(Wc(:, :, filterNum), 2);
end

meanPoolingFilter = ones(poolDim)/poolDim ^ 2;
poolingIndex = 1 : poolDim : size(conv2(conv2(images(:, :, 1), Wc_rotated(:, :, 1), 'valid'), meanPoolingFilter, 'valid'), 1);

for imageNum = 1 : numImages
    image = images(:, :, imageNum);
    for filterNum = 1 : numFilters
        %         filter = Wc_rotated(:, :, filterNum);
        %         filteredImage = conv2(image, filter, 'valid');
        
        convolveImage = conv2(image, Wc_rotated(:, :, filterNum), 'valid') + bc(filterNum);
        
        switch activationType
            case 'relu'
                convolveImage = max(convolveImage, 0); % relu
            case 'sigmoid'
                convolveImage = sigmoid(convolveImage); % sigmoid
        end
        % the main reason why I was wrong !
        % cannot directly assign a calculation to a different dimension matrix !
        % for example, resulted 2D cannnot be directly assign to 4D matrix
        activations(:, :, filterNum, imageNum) = convolveImage;
        pooledImage = conv2(convolveImage, meanPoolingFilter, 'valid');
        activationsPooled(:, :, filterNum, imageNum) = pooledImage(poolingIndex, poolingIndex);
    end
end

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%

% (numClasses x hiddenSize) * (hiddenSize x numImages)
activationsSoftmax = Wd * activationsPooled + repmat(bd, 1, numImages);
% activationsSoftmax = bsxfun(@plus, Wd * activationsPooledReshaped, bd);
activationsSoftmax = bsxfun(@minus, activationsSoftmax, max(activationsSoftmax));
activationsSoftmax = exp(activationsSoftmax);
probs = bsxfun(@rdivide, activationsSoftmax, sum(activationsSoftmax));

%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%

labels = full(sparse(labels,1:numImages,1));

cost = -sum(sum(labels .* log(probs)));
cost = cost / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%

% Backpropagate through the softmax layer
errorsSoftmax = probs - labels;
errorsSoftmax = errorsSoftmax / numImages;

% Backpropagate through the mean pooling layer
errorsPooled = Wd' * errorsSoftmax;
errorsPooled = reshape(errorsPooled, [], outputDim, numFilters, numImages);

errorsPooling = zeros(convDim, convDim, numFilters, numImages);
% unpoolingFilter = ones(poolDim);
% unpoolingFilter = unpoolingFilter / poolDim ^ 2;
% the upsampling filter is still meanPooling filter
for imageNum = 1:numImages
    % for imageNum = 1:numImages
    for filterNum = 1:numFilters
        error = errorsPooled(:, :, filterNum, imageNum);
        errorsPooling(:, :, filterNum, imageNum) = kron(error, meanPoolingFilter);
        
        %         errorsPooling(:, :, filterNum, imageNum) = kron(errorsPooled(:, :, filterNum, imageNum), unpoolingFilter);
    end
end

switch activationType
    case 'relu'
        errorsConvolution = errorsPooling .* (activations > 0); % relu derivative = x > 1
    case 'sigmoid'
        errorsConvolution = errorsPooling .* activations .* (1 - activations); % sigmoid derivative = x .* (1 - x)
end


%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%

% Gradient of the softmax layer
Wd_grad = errorsSoftmax * activationsPooled';
bd_grad = sum(errorsSoftmax, 2);

% for filterNum = 1 : numFilters
%     Wc_grad_temp = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
%     for imageNum = 1: numImages
%         error = errorsConvolution(:, :, filterNum, imageNum);
%         Wc_grad_temp = Wc_grad(:, :, filterNum) + conv2(images(:, :, imageNum), error, 'valid');
%     end
%     Wc_grad(:, :, filterNum) = Wc_grad_temp;
% end

for filterNum = 1 : numFilters
    for imageNum = 1 : numImages
        e = errorsConvolution(:, :, filterNum, imageNum);
        %         e = errorsPooling(:, :, filterNum, imageNum);
        errorsConvolution(:, :, filterNum, imageNum) = rot90(e, 2);
        
        %         errorsPooling(:, :, filterNum, imageNum) = rot90(errorsPooling(:, :, filterNum, imageNum), 2);
    end
end

for filterNum = 1 : numFilters
    Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
    for imageNum = 1 : numImages
        %                 image = images(:, :, imageNum);
        %                 error = errorsPooling(:, :, filterNum, imageNum);
        %         %         Wc_grad(:, :, filterNum) = Wc_grad(:, :, filterNum) + conv2(image, error, 'valid');
        %         Wc_gradFilter(:, :, imageNum) = conv2(image, error, 'valid');
        %         Wc_gradFilter(:, :, imageNum) = conv2(images(:, :, imageNum), errorsPooling(:, :, filterNum, imageNum), 'valid');
        
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), errorsConvolution(:, :, filterNum, imageNum), 'valid');
        %         Wc_gradFilter = Wc_gradFilter + conv2(image, error, 'valid');
    end
    %     Wc_grad(:, :, filterNum) = sum(Wc_gradFilter, 3) / numImages + regularization;
    Wc_grad(:, :, filterNum) = Wc_gradFilter;
end

bc_grad = reshape(sum(sum(sum(errorsConvolution,4),2),1),[],numFilters);


%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
