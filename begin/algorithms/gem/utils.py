import torch
from qpth.qp import QPFunction

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    with torch.no_grad():
        t = memories.shape[0]
        _eye = torch.eye(t, device=gradient.device)
        G = -_eye
        h = torch.zeros(t, device=gradient.device) - margin
        
        while True:
            try:
                Q = memories @ memories.t()
                Q = (0.5 * (Q + Q.t()) + _eye * eps)
                p = (memories @ gradient)
                v = QPFunction(verbose=False)(Q, p, G, h, torch.zeros(0, device=gradient.device), torch.zeros(0, device=gradient.device))[0]
                break
            except:
                eps = eps * 10.
                
        x = (v @ memories) + gradient
        return x.detach()
    