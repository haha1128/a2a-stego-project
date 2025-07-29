import torch

def differential_based_recombination(prob, indices):
    bins = []
    
    prob, sorted_indices = torch.sort(prob, descending=False)
    indices = indices[sorted_indices]

    mask = prob > 0
    prob_nonzero = prob[mask]
    indices_nonzero = indices[mask]
 
    diff = torch.cat((prob_nonzero[:1], torch.diff(prob_nonzero, n=1)))
    n = len(prob_nonzero)

    weights = torch.arange(n, 0, -1, device = prob.device) 

    diff_positive = diff > 0

    prob_new = diff[diff_positive] * weights[diff_positive] 
    bins = torch.arange(n, device = prob.device)[diff_positive]

    return indices_nonzero, bins, prob_new

def binary_based_recombination(prob, indices, precision=52):

    mask = prob > 0
    prob_nonzero = prob[mask]
    indices_nonzero = indices[mask]

    scale = 2 ** precision
    scaled_probs = (prob_nonzero * scale).to(torch.int64)

    masks = (1 << torch.arange(precision)).to(scaled_probs.device)  
    masked_probs = scaled_probs.unsqueeze(1) & masks

    nonzero_mask_indices = masked_probs > 0

    bins = [indices_nonzero[nonzero_mask_indices[:, k]] for k in range(precision)]
    bin_sizes = torch.tensor([len(g) for g in bins], dtype=torch.long)

    power_i = (2 ** (-torch.arange(precision, 0, -1).float()))  

    prob_new = bin_sizes * power_i
    
    return bins, prob_new 