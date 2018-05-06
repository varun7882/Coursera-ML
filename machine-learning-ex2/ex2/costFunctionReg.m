function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sgOfh=sigmoid(X*theta);
tmp1=y'*log(sgOfh);
tmp2=(1-y)'*log(1-sgOfh);
tmp=tmp1+tmp2;
J=-(tmp/m);
grad = m^(-1) * ((sgOfh-y)'*X)'; 
rg=sum(theta.^2);
rg=rg-theta(1,1)^2;
J=J+((lambda/(2*m))*(rg));
shift_theta = theta(2:size(theta));
rg = [0;shift_theta];
grad=grad+((lambda/(m))*(rg));


% =============================================================

end
