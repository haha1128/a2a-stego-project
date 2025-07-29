# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
from cython.operator cimport dereference as deref
from libc.math cimport log2
from libcpp cimport nullptr
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.memory cimport shared_ptr, make_shared

import time
import random
import numpy as np
from PIL import Image
import torch
from scipy.stats import entropy
from tqdm import tqdm


## Classes & Structures
# Nodes of Huffman tree 
cdef struct Node:
    double prob
    shared_ptr[Node] left
    shared_ptr[Node] right
    int index
    # >=0 - index
    # -1 - None
    int search_path
# 0  - this node
# -1 - in left subtree
# 1  - in right subtree
# 9  - unknown


cdef inline bint is_leaf(shared_ptr[Node] node_ptr):
    return deref(node_ptr).index != -1

# Sampling (Encoding) results and statistics for single time step
cdef struct CySingleEncodeStepOutput:
    int sampled_index
    int n_bits

# Decoding results for single time step
cdef struct CySingleDecodeStepOutput:
    string message_decoded_t

## Utils
# Building a Huffman tree
cdef shared_ptr[Node] create_huffman_tree(list indices, list probs, int search_for):
    # Returns a pointer to the root node of the Huffman tree
    # if `search_for == -1`, we don't need to initialize the `search_path` of any Node object
    cdef:
        int sz = len(indices)
        int i, search_path
        double prob
        shared_ptr[Node] node_ptr, first, second, ans
        queue[shared_ptr[Node]] q1, q2

    for i in range(sz - 1, -1, -1):
        # search_path = 0 if search_for == indices[i] else 9
        if search_for == indices[i]:
            search_path = 0
        else:
            search_path = 9
        node_ptr = make_shared[Node](
            Node(probs[i], shared_ptr[Node](nullptr), shared_ptr[Node](nullptr), indices[i], search_path))
        q1.push(node_ptr)

    while q1.size() + q2.size() > 1:
        # first
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            first = q1.front()
            q1.pop()
        elif q1.empty():
            first = q2.front()
            q2.pop()
        elif q2.empty():
            first = q1.front()
            q1.pop()
        else:
            first = q2.front()
            q2.pop()

        # second
        if not q1.empty() and not q2.empty() and deref(q1.front()).prob < deref(q2.front()).prob:
            second = q1.front()
            q1.pop()
        elif q1.empty():
            second = q2.front()
            q2.pop()
        elif q2.empty():
            second = q1.front()
            q1.pop()
        else:
            second = q2.front()
            q2.pop()

        # merge
        prob = deref(first).prob + deref(second).prob
        search_path = 9
        if deref(first).search_path != 9:
            search_path = -1
        elif deref(second).search_path != 9:
            search_path = 1
        q2.push(make_shared[Node](Node(prob, first, second, -1, search_path)))

    if not q2.empty():
        ans = q2.front()
    else:
        ans = q1.front()
    return ans

## Steganography process - single time step
# Sampling (Encoding) - single time step
cdef CySingleEncodeStepOutput cy_encode_step(list indices, list probs, string message_bits, int bit_index, mask_generator, int precision):
    cdef:
        int sampled_index, n_bits = 0
        double prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, -1)
        vector[int] path_table = [-1, 1]
        int len_message_bits = len(message_bits)

    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = mask_generator.generate_random(n = precision)
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1

        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1



        if path_table[message_bits[n_bits+bit_index] - 48] == 1:
            node_ptr = deref(node_ptr).right
        else:
            node_ptr = deref(node_ptr).left


        if path_table[0] != path_table[1]:
            n_bits += 1
    # print(deref(node_ptr).index)
    sampled_index = deref(node_ptr).index

    return CySingleEncodeStepOutput(sampled_index, n_bits)

cdef CySingleEncodeStepOutput cy_baseline_encode_step(list indices, list probs, string message_bits, int bit_index, mask_generator, int precision):
    cdef:
        int sampled_index, n_bits = 0, capacity, capacity_upper_bound, i
        double ptr, ptr_i, rotate_step_size
        int len_message_bits = len(message_bits)
    
    probs_cumsum = torch.tensor(probs).cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    # Determine capacity
    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1

    tbl = {}  # message bits -> token_index
    ptr = mask_generator.generate_random(n = precision)

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0**-capacity
        is_available = True
        tbl_new = {}
        for i in range(int(2**capacity)):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1
            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]
            if index in tbl_new.values():
                is_available = False
                break
            tbl_new[i] = index
        if not is_available:
            break
        tbl = tbl_new
        n_bits = capacity
        capacity += 1
    if n_bits < 1:
        sampled_index = indices[(ptr >= interval_begin).nonzero()[-1].item()]
    else:
        cur_message_bits_decimal = 0
        base = 1
        for d in range(n_bits - 1, -1, -1):
            if message_bits[bit_index+d] == b'1':
                cur_message_bits_decimal += base
            base *= 2
        sampled_index = tbl[cur_message_bits_decimal]

    return CySingleEncodeStepOutput(sampled_index, n_bits)





cdef CySingleDecodeStepOutput cy_decode_step(list indices, list probs, int stego_t, mask_generator, int precision):
    # Decode step
    cdef:
        string message_decoded_t
        double prob_sum, ptr, ptr_0, ptr_1, partition
        shared_ptr[Node] node_ptr = create_huffman_tree(indices, probs, stego_t)
        vector[int] path_table = vector[int](2)
        map[int, string] path_table_swap

    while not is_leaf(node_ptr):  # non-leaf node
        prob_sum = deref(node_ptr).prob
        ptr = mask_generator.generate_random(n = precision)
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = deref(deref(node_ptr).left).prob

        # path_table[0] = -1 if (ptr_0 < partition) else 1
        if ptr_0 < partition:
            path_table[0] = -1
        else:
            path_table[0] = 1
        # path_table[1] = -1 if (ptr_1 < partition) else 1
        if ptr_1 < partition:
            path_table[1] = -1
        else:
            path_table[1] = 1

        if path_table[0] != path_table[1]:  # can embed 1 bit
            if deref(node_ptr).search_path == 9:  # fail to decode
                message_decoded_t = b''
                break

            if path_table[0] == -1:
                path_table_swap[-1] = b'0'
                path_table_swap[1] = b'1'
            else:
                path_table_swap[-1] = b'1'
                path_table_swap[1] = b'0'
            message_decoded_t += path_table_swap[deref(node_ptr).search_path]

            # walk
            if deref(node_ptr).search_path == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right
        else:
            if path_table[0] == -1:
                node_ptr = deref(node_ptr).left
            else:
                node_ptr = deref(node_ptr).right

    if deref(node_ptr).search_path != 0:  # cannot reach a leaf node
        message_decoded_t = b''
    return CySingleDecodeStepOutput(message_decoded_t)

cdef CySingleDecodeStepOutput cy_baseline_decode_step(list indices, list probs, int stego_t, mask_generator, int precision):
    # Decode step
    cdef:
        int capacity, capacity_upper_bound, n_bits = 0
        string message_decoded_t
        double ptr
    probs_cumsum = torch.tensor(probs).cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)

    # Determine capacity
    capacity = int(log2(1 / probs[0]))
    capacity_upper_bound = capacity + 1

    tbl = {}  # message bits -> token_index
    ptr = mask_generator.generate_random(n = precision)

    while capacity <= capacity_upper_bound:
        if capacity == 0:
            capacity += 1
            continue
        rotate_step_size = 2.0**-capacity
        is_available = True
        tbl_new = {}
        for i in range(int(2**capacity)):
            ptr_i = ptr + i * rotate_step_size
            if ptr_i >= 1.0:
                ptr_i -= 1
            index_idx = (ptr_i >= interval_begin).nonzero()[-1].item()
            index = indices[index_idx]
            if index in tbl_new.values():
                is_available = False
                break
            tbl_new[i] = index
        if not is_available:
            break
        tbl = tbl_new
        n_bits = capacity
        capacity += 1
    if n_bits < 1:
        message_decoded_t = b''
    else:
        if stego_t not in tbl.values():  # Error
            message_decoded_t = b''
        tbl_swapped = dict(zip(tbl.values(), tbl.keys()))  # token_index -> message bits
        message_decoded_t = bin(tbl_swapped[stego_t])[2:].zfill(n_bits)
    return CySingleDecodeStepOutput(message_decoded_t)



def Discop_encoder(prob, indices, bit_stream, bit_index, mask_generator, precision):
    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]    
    
    prob = prob.tolist()
    indices = indices.tolist()

    s = cy_encode_step(indices, prob, bit_stream, bit_index, mask_generator, precision)


    return torch.tensor(s.sampled_index).view(1,1).cuda(),s.n_bits

def Discop_base_encoder(prob, indices, bit_stream, bit_index, mask_generator, precision):
    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]    
    
    prob = prob.tolist()
    indices = indices.tolist()

    s = cy_baseline_encode_step(indices, prob, bit_stream, bit_index, mask_generator, precision)


    return torch.tensor(s.sampled_index).view(1,1).cuda(),s.n_bits


def Discop_decoder(prob, indices, prev, mask_generator, precision):
    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]    

    prob = prob.tolist()
    indices = indices.tolist()

    return cy_decode_step(indices, prob, prev, mask_generator, precision).message_decoded_t

def Discop_base_decoder(prob, indices, prev, mask_generator, precision):
    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]    

    prob = prob.tolist()
    indices = indices.tolist()

    return cy_baseline_decode_step(indices, prob, prev, mask_generator, precision).message_decoded_t


