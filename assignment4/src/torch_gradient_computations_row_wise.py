import torch
import numpy as np
# assumes X has size tau x d, h0 has size 1 x m, etc
def ComputeGradsWithTorch(X, y, h0, W, U, V, b, c):

    tau = X.shape[0]

    Xt = torch.from_numpy(X).double()
    ht = torch.from_numpy(h0).double()
    torch_network = {}
    for kk in ['W', 'U', 'V', 'b', 'c']:
        torch_network[kk] = torch.tensor(locals()[kk],dtype=torch.float64, requires_grad=True)


    ## give informative names to these torch classes        
    apply_tanh = torch.nn.Tanh()
    apply_softmax = torch.nn.Softmax(dim=1)
        
    # create an empty tensor to store the hidden vector at each timestep
    Hs = torch.empty(X.shape[0], h0.shape[1], dtype=torch.float64)
  
    hprev = ht
    for t in range(tau):

        #### BEGIN your code ######

        # Code to apply the RNN to hprev and Xt[t:t+1, :] to compute the hidden scores "Hs" at timestep t
        # (ie equations (1,2) in the assignment instructions)
        # Store results in Hs
        h = apply_tanh(Xt[t:t+1, :] @ torch_network['U'] + hprev @ torch_network['W'] + torch_network['b'])
        Hs[t:t+1, :] = h
        hprev = h # update hprev for the next time step
        # Don't forget to update hprev!
        
        #### END of your code ######            

    Os = torch.matmul(Hs, torch_network['V']) + torch_network['c']
    P = apply_softmax(Os)    
    
    # compute the loss
    loss = torch.mean(-torch.log(P[np.arange(tau), y])) # use this line if storing inputs row-wise

    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()
    print("loss:", loss.item())
    # extract the computed gradients and make them numpy arrays
    grads = {}
    for kk in ['W', 'U', 'V', 'b', 'c']:
        grads[kk] = torch_network[kk].grad.numpy()

    return grads
