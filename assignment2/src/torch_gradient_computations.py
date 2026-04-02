import torch
import numpy as np

# def ComputeGradsWithTorch(X, y, network_params):
    
#     Xt = torch.from_numpy(X)

#     L = len(network_params['W'])

#     # will be computing the gradient w.r.t. these parameters    
#     W = [None] * L
#     b = [None] * L    
#     for i in range(len(network_params['W'])):
#         W[i] = torch.tensor(network_params['W'][i], requires_grad=True)
#         b[i] = torch.tensor(network_params['b'][i], requires_grad=True)        

#     ## give informative names to these torch classes        
#     apply_relu = torch.nn.ReLU()
#     apply_softmax = torch.nn.Softmax(dim=0)

#     #### BEGIN your code ###########################
    
#     # Apply the scoring function corresponding to equations (1-3) in assignment description 
#     # If X is d x n then the final scores torch array should have size 10 x n 

#     #### END of your code ###########################            

#     # apply SoftMax to each column of scores     
#     P = apply_softmax(scores)
    
#     # compute the loss
#     n = X.shape[1]
#     loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
#     # compute the backward pass relative to the loss and the named parameters 
#     loss.backward()

#     # extract the computed gradients and make them numpy arrays 
#     grads = {}
#     grads['W'] = [None] * L
#     grads['b'] = [None] * L
#     for i in range(L):
#         grads['W'][i] = W[i].grad.numpy()
#         grads['b'][i] = b[i].grad.numpy()

#     return grads

def ComputeGradsWithTorch(X, y, W1, b1, W2, b2, reg: int = 0):

    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W1 = torch.tensor(W1, requires_grad=True)
    b1 = torch.tensor(b1, requires_grad=True)    
    
    W2 = torch.tensor(W2, requires_grad=True)
    b2 = torch.tensor(b2, requires_grad=True)    
    N = X.shape[1]

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)


    # apply softmax to each column of scores
    tmp = apply_relu(torch.matmul(W1, Xt) + b1)
    scores = torch.matmul(W2, tmp) + b2
    P = apply_softmax(scores)
    
    ## compute the loss

    loss = torch.mean(-torch.log(P[y, np.arange(N)]))    
    print(f"Loss computed with PyTorch: {loss.item():.12f}")
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W1'] = W1.grad.numpy() + 2 * reg * W1.detach().numpy() # add L2 regularization gradient
    grads['b1'] = b1.grad.numpy()
    grads['W2'] = W2.grad.numpy() + 2 * reg * W2.detach().numpy()
    grads['b2'] = b2.grad.numpy()
    return grads    
