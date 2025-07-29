import torch
from .utils import HuffmanCoding
import numpy as np
from configparser import ConfigParser
import prtpy
import hashlib
import hmac
import random
import time
import math

from ..discop import Discop_encoder,Discop_base_encoder


def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res


def msb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in strlist]

#lsb
# e.g. [0, 1, 1, 1] looks like 1110=14
def lsb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def lsb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0
        self.test = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        self.test+=1
        xs = np.zeros(n, dtype=bool)
        for i in range(0,n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs
    
    def generate_random(self, n):
        
        xs = self.generate_bits(n)
        def binary_array_to_float(bin_array):
            # Convert the binary array to a decimal integer
            decimal_value = 0
            for bit in bin_array:
                decimal_value = (decimal_value << 1) | bit
            
            # Calculate the maximum possible value (all bits are 1)
            max_value = (1 << len(bin_array)) - 1
            
            # Return a float between 0 and 1
            return decimal_value / max_value
        return binary_array_to_float(xs)


def AC_encoder(prob, indices, bit_stream, bit_index, cur_interval, precision):

    prob, sorted_indices = torch.sort(prob, descending=True)
    indices = indices[sorted_indices]

    cur_int_range = cur_interval[1] - cur_interval[0]  # The size of the interval, e.g., 2^52
    cur_threshold = 1 / cur_int_range  # How large each sub-interval is
    if prob[-1] < cur_threshold:
        k = max(2, (prob < cur_threshold).nonzero()[0].item())
        prob = prob[:k]
        indices = indices[:k]

    prob = prob / prob.sum()  # Linear normalization after truncation
    prob = prob.double()
    prob *= cur_int_range  # Convert probabilities to the number of intervals
    prob = prob.round().long()  # Round to the nearest integer, probabilities described by the number of intervals

    cum_probs = prob.cumsum(0)  # Cumulative sum of probabilities, describing the distribution function in terms of intervals
    overfill_index = (cum_probs > cur_int_range).nonzero()
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]  # Remove the last probability if it overfills
    cum_probs += cur_int_range - cum_probs[-1]  # Adjust the CDF to sum to the total range

    cum_probs += cur_interval[0]  # Start the CDF from the left of the current interval

    message_bits = bit_stream[bit_index: bit_index + precision]  # Take 52 bits, used to lock a position, not to be encoded
    message_bits = [int(_) for _ in message_bits]
    message_idx = msb_bits2int(message_bits)
    selection = (cum_probs > message_idx).nonzero()[0].item()

    new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[
        0]  # New left boundary; if the first word is chosen (selection=0), the left boundary doesn't move
    new_int_top = cum_probs[selection]

    new_int_bottom_bits_inc = list(msb_int2bits(new_int_bottom, precision))  # Binary lower bound
    new_int_top_bits_inc = list(msb_int2bits(new_int_top - 1, precision))  # Binary upper bound

    num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

    new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # New binary interval
    new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

    cur_interval[0] = msb_bits2int(new_int_bottom_bits)  # New interval
    cur_interval[1] = msb_bits2int(new_int_top_bits) + 1  # +1 here because upper bound is exclusive
    prev = indices[selection].view(1, 1)  # A number representing which word was chosen
    return cur_interval, prev, num_bits_encoded


def ac_encoder(prob, indices, bit_stream, bit_index,  cur_interval,precision):
    return AC_encoder(prob, indices, bit_stream, bit_index, cur_interval, precision)
    
def discop_encoder(alg, prob, indices, bit_stream, bit_index,mask_generator, precision):
    if alg.lower() == "discop":
        return Discop_encoder(prob, indices, bit_stream, bit_index, mask_generator, precision) 
    if alg.lower() == "discop_base":
        return Discop_base_encoder(prob, indices, bit_stream, bit_index, mask_generator, precision)
    raise ValueError("no such algorithm")