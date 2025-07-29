import torch

from .recombination import binary_based_recombination, differential_based_recombination
from .uni_stego import uni_binary_enc, uni_cyclic_shift_enc

### time-step encoding

def differential_based_encoder(prob, indices, bit_stream, bit_index, PRG, precision):
   
    #Probability Recombination Module
    indices_nonzero, bins, prob_new = differential_based_recombination(prob, indices)
    prob_new = prob_new/prob_new.sum()

    #Bin Sampling
    random_p = PRG.generate_random(n = precision)
    cdf = torch.cumsum(prob_new, dim=0)
    bin_indice = torch.searchsorted(cdf, random_p).item()
    bin = indices_nonzero[bins[bin_indice]:]

    #Uniform Steganography Module
    idx,bits = uni_cyclic_shift_enc(bit_stream=bit_stream[bit_index:], n = len(bin), PRG = PRG, precision=precision)
    
    num = len(bits)
    prev = bin[idx].view(1,1)

    return prev, num

def binary_based_encoder(prob, indices, bit_stream, bit_index, PRG, precision):
    if (prob[0] == 1):
        return indices[0].view(1,1), 0
    
    #Probability Recombination Module
    bins, prob_new = binary_based_recombination(prob, indices, precision)

    #Bin Sampling
    random_p = PRG.generate_random(n = precision)
    cdf = torch.cumsum(prob_new, dim=0)
    bin_indice = torch.searchsorted(cdf, random_p).item()
    bin = bins[bin_indice]

    #Uniform Steganography Module
    idx,bits = uni_cyclic_shift_enc(bit_stream=bit_stream[bit_index:],n = len(bin), PRG = PRG, precision=precision)
    
    num = len(bits)
    prev = bin[idx].view(1,1)
    
    return prev, num

def stability_based_encoder(prob, indices, bit_stream, bit_index, PRG, precision):

    # Ensure all tensors are on the same device
    device = prob.device
    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]

    get_idx_r = lambda arr, x: torch.searchsorted(arr, x, side="right")
    
    def _sample_bin(p_sum, q_sum, t):
        """
        Parameters
        -----
            p: cumsum of prabability array
            q: cumsum of split array
            t: random number from `Uniform[0, 1)`

        Return
        -----
            An array of indices for the sampled bin.
        """
        assert torch.allclose(p_sum[-1], q_sum[-1])
        _zero = torch.tensor([0], device=device)
        q_sum2 = torch.concatenate([_zero, q_sum])
        
        i = get_idx_r(q_sum, t)

        s = t - q_sum2[i]

        l = q_sum2[:-1] + s
        
        l = l[l < q_sum]

        return get_idx_r(p_sum, l)

    def sample_method2(p, t, p2_max=1):

        p, _ = torch.sort(p, descending=True)
        q2 = (1 - p2_max) * p[1] + p2_max * min(p[0], 1 - p[0] - 1e-8)
        p_sum = torch.cumsum(p,dim = 0)

        q2s = p_sum[0] + q2
        q_sum = torch.concatenate((torch.tensor([p_sum[0], q2s], device=device), p_sum[p_sum > q2s]), axis=0)

        return _sample_bin(p_sum, q_sum, t)

    random_p = PRG.generate_random(n = precision)
    bin = indices[sample_method2(prob, random_p)]

    #Uniform Steganography Module
    idx,bits = uni_cyclic_shift_enc(bit_stream=bit_stream[bit_index:],n = len(bin), PRG = PRG, precision=precision)

    num = len(bits)
    prev = bin[idx].view(1,1)

    return prev, num


def encoder(alg, prob, indices, bit_stream, bit_index,PRG, precision):
    if alg.lower() == "differential_based":
        return differential_based_encoder(prob, indices, bit_stream, bit_index,PRG, precision)
    if alg.lower() == "binary_based":
        return binary_based_encoder(prob, indices, bit_stream, bit_index,PRG, precision)
    if alg.lower() == "stability_based":
        return stability_based_encoder(prob, indices, bit_stream, bit_index,PRG, precision)
    raise ValueError("no such scheme") 