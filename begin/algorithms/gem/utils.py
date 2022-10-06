import numpy as np
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
        
        Q = memories @ memories.t()
        Q = (0.5 * (Q + Q.t()) + _eye * eps)
        p = (memories @ gradient)
        
        v = QPFunction(verbose=False)(Q, p, G, h, torch.zeros(0, device=gradient.device), torch.zeros(0, device=gradient.device))[0]
        x = (v @ memories) + gradient
        return x.detach()
    
def project2cone2_numpy(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().double().numpy()
    gradient_np = gradient.cpu().double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    return x
    