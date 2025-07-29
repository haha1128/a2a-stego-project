import os
import random
import json
import math
import torch
from tqdm.autonotebook import tqdm
from .baselines.encode import ac_encoder, discop_encoder, DRBG
from .baselines.decode import ac_decoder, discop_decoder
from .artifacts_baselines.encode import encoder as artifacts_encoder
from .artifacts_baselines.decode import decoder as artifacts_decoder
from .artifacts_baselines.utils import DRBG as ArtifactsDRBG
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import config
from modules.logging.logging_mannager import LoggingMannager


discop_alg = ['discop','discop_base']
artifacts_alg = ['differential_based', 'binary_based', 'stability_based']

logger = LoggingMannager.get_logger(__name__)

def prompt_template(prompt_text, model, tokenizer, mode='chat', role='user'):
    """
    Processes the input prompt and generates a corresponding token sequence based on the specified mode.
    """
    if mode == 'generate':
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
        return input_ids
    elif mode == 'chat':
        messages = [{"role": role, "content": prompt_text}]
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        return tokenized_chat
    else:
        raise ValueError("no such mode")

def calculate_entropy(prob):
    """Calculates the entropy of a probability distribution."""
    prob_np = prob.detach().cpu().numpy()
    prob_np = prob_np[prob_np > 0]  # Avoid log(0)
    return -np.sum(prob_np * np.log2(prob_np))

def set_seed(seed):
    """Sets the random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def _get_mask_generator(algorithm:str,kwargs:dict):
    """Gets the mask generator (for provably secure algorithms)."""
    if algorithm in discop_alg:
        input_key = bytes.fromhex(kwargs["input_key"])
        sample_seed_prefix = bytes.fromhex(kwargs["sample_seed_prefix"])
        input_nonce = bytes.fromhex(kwargs["input_nonce"])
        return DRBG(input_key, sample_seed_prefix + input_nonce)
    elif algorithm in artifacts_alg:
        input_key = bytes.fromhex(kwargs["input_key"])
        sample_seed_prefix = bytes.fromhex(kwargs["sample_seed_prefix"])
        input_nonce = bytes.fromhex(kwargs["input_nonce"])
        return ArtifactsDRBG(input_key, sample_seed_prefix + input_nonce)
    return None
    
def encrypt(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,algorithm:str,bit_stream:str, prompt_text:str):
    """
    Encryption function: embeds a bitstream into the generated text.
    
    Args:
        bit_stream (str): The '01' string to be embedded.
        prompt_text (str): The context prompt.
        
    Returns:
        tuple: (steganographic text, number of bits actually embedded, sequence of generated token IDs)
    """
    set_seed(config.ALGORITHM_CONFIG["seed"])
    algorithm_kwargs = config.ALGORITHM_CONFIG[algorithm]
    LLM_CONFIG = config.LLM_CONFIG
    max_new_tokens = LLM_CONFIG["max_new_tokens"]
    topk = LLM_CONFIG["topk"]

    mask_generator = _get_mask_generator(algorithm,algorithm_kwargs)
    original_bit_length = len(bit_stream)
    required_bits = max_new_tokens * math.log2(tokenizer.vocab_size)
    bit_index = 0
    if len(bit_stream[bit_index:]) <= required_bits:
        bit_stream_shuffle = np.random.randint(high=2, low=0, size=(1, 100000)).tolist()[0]
        random.shuffle(bit_stream_shuffle)
        bit_stream += "".join([str(b) for b in bit_stream_shuffle])

    with torch.no_grad():
        input_ids = prompt_template(prompt_text, model, tokenizer, 
                                    mode=LLM_CONFIG["prompt_template"], role=LLM_CONFIG["role"])
        x = input_ids
        stega_sentence = []
        stega_bit = []
        total_bits_embedded = 0
        
        # Special initialization for the AC algorithm
        if algorithm.lower() in ["ac"]:
            max_val = 2 ** algorithm_kwargs["precision"]
            cur_interval = [0, max_val]
           
            
        past_key_values = None
        
        for i in tqdm(range(max_new_tokens), desc="Generating steganographic text"):
            if tokenizer.eos_token_id in stega_sentence:
                break
                
            # Get the conditional probability distribution
            output = model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            log_prob = output.logits[:, -1, :].clone()  # Use clone to avoid reference
            log_prob -= log_prob.max()
            prob = torch.exp(log_prob).reshape(-1)
            prob = prob / prob.sum()
            
            # Clean up output to free up GPU memory
            del output
            del log_prob
            
            # Preprocess logits
            prob, indices = prob.sort(descending=True)
            mask = prob > 0
            prob = prob[mask]
            indices = indices[mask]
            prob = prob[:topk]
            indices = indices[:topk]
            prob = prob / prob.sum()
            
            # Encode according to the algorithm
            if algorithm.lower() in ["ac"]:
                cur_interval, prev, num_bits_encoded = ac_encoder(
                    prob, indices, bit_stream, bit_index, 
                    cur_interval,algorithm_kwargs["precision"])
            elif algorithm.lower() in discop_alg:
                prev, num_bits_encoded = discop_encoder(
                    algorithm, prob, indices, bit_stream, bit_index,
                    mask_generator, algorithm_kwargs["precision"])
            elif algorithm.lower() in artifacts_alg:
                prev, num_bits_encoded = artifacts_encoder(
                    algorithm, prob, indices, bit_stream, bit_index,
                    mask_generator, algorithm_kwargs["precision"])
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            if int(prev) == tokenizer.eos_token_id:
                break
                
            stega_sentence.append(int(prev))
            x = prev.reshape(1, 1)
            stega_bit.append(bit_stream[bit_index:bit_index + num_bits_encoded])
            total_bits_embedded += num_bits_encoded
            bit_index += num_bits_encoded
            
            # Clean up temporary tensors
            del prob, indices, mask, prev
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    # Remove the EOS token
    if tokenizer.eos_token_id in stega_sentence:
        stega_sentence.remove(tokenizer.eos_token_id)
        
    stega_text = tokenizer.decode(stega_sentence)
    actual_embedded_bits = min(total_bits_embedded, original_bit_length)
    logger.info(f"Embedded bits: {bit_stream[:actual_embedded_bits]}")
    
    return stega_text, actual_embedded_bits, stega_sentence
    
def decrypt(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,algorithm:str,stego_text:str, prompt_text:str):
    """
    Decryption function: extracts a bitstream from steganographic text.
    
    Args:
        stego_text (str): The steganographic text.
        prompt_text (str): The context prompt used during generation (must be the same as during encryption).
        max_tokens (int): The maximum number of tokens to decode; if None, the setting from initialization is used.
        
    Returns:
        tuple: (the full extracted bitstream, a list of bitstrings extracted for each token, the sequence of parsed token IDs)
    """
    set_seed(config.ALGORITHM_CONFIG["seed"])
    algorithm_kwargs = config.ALGORITHM_CONFIG[algorithm]
    LLM_CONFIG = config.LLM_CONFIG
    max_tokens = LLM_CONFIG["max_new_tokens"]
    topk = LLM_CONFIG["topk"]
    
        
    mask_generator = _get_mask_generator(algorithm,algorithm_kwargs)
    
    with torch.no_grad():
        input_ids = prompt_template(prompt_text, model, tokenizer,
                                    mode=LLM_CONFIG["prompt_template"], role=LLM_CONFIG["role"])
        
        # Special initialization for the AC algorithm
        if algorithm.lower() in ["ac"]:
            max_val = 2 ** algorithm_kwargs["precision"]
            cur_interval = [0, max_val]
            
        full_bits = ""
        past_key_values = None
        tokens = []
        tokens_bits = []
        
        # Concatenate the prompt and the steganographic text
        full_ids = torch.cat((
            input_ids,
            tokenizer.encode(stego_text, add_special_tokens=False, return_tensors="pt").to(model.device)
        ), dim=1)
        
        # Iterate through each token of the steganographic text
        for i in tqdm(range(len(input_ids[0]), min(len(full_ids[0]), max_tokens + len(input_ids[0]))),
                        desc="Extracting hidden information"):
            
            # Get the conditional probability distribution
            output = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            log_prob, past_key_values = output.logits, output.past_key_values
            log_prob = log_prob[0, -1, :]
            log_prob -= log_prob.max()
            prob = torch.exp(log_prob).reshape(-1)
            prob = prob / prob.sum()
            
            # Preprocess logits
            prob, indices = prob.sort(descending=True)
            mask = prob > 0
            prob = prob[mask]
            indices = indices[mask]
            prob = prob[:topk]
            indices = indices[:topk]
            prob = prob / prob.sum()
            
            embed_id = full_ids[0][i].item()
            tokens.append(embed_id)
            
            try:
                # Decode according to the algorithm
                if algorithm.lower() in ["ac"]:
                    cur_interval, extract_bits = ac_decoder(
                        prob, indices, embed_id, cur_interval, algorithm_kwargs["precision"])
                elif algorithm.lower() in discop_alg:
                    extract_bits = discop_decoder(
                        algorithm, prob, indices, embed_id, 
                        mask_generator, algorithm_kwargs["precision"])
                elif algorithm.lower() in artifacts_alg:
                    extract_bits = artifacts_decoder(
                        algorithm, prob, indices, embed_id, 
                        mask_generator, algorithm_kwargs["precision"])
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
            except:
                extract_bits = ""
                
            input_ids = full_ids[0][i].reshape(1, 1)
            full_bits += extract_bits
            tokens_bits.append(extract_bits)
            
    return full_bits, tokens_bits, tokens
def generate_text(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,prompt_text:str):
    """
    Generates text normally, without steganography.
    
    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        prompt_text (str): The input prompt text.
        
    Returns:
        tuple: (the generated text, the sequence of generated token IDs)
    """
    set_seed(config.ALGORITHM_CONFIG["seed"])
    LLM_CONFIG = config.LLM_CONFIG
    max_new_tokens = LLM_CONFIG["max_new_tokens"]
    topk = LLM_CONFIG["topk"]
    with torch.no_grad():
        input_ids = prompt_template(prompt_text, model, tokenizer, 
                                    mode=LLM_CONFIG["prompt_template"], role=LLM_CONFIG["role"])
        
        generated_tokens = []
        past_key_values = None
        x = input_ids
        for i in tqdm(range(max_new_tokens), desc="Generating normal text"):
            # Get model output
            output = model(input_ids=x, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :]
            
            # Apply top-k sampling
            if topk is not None and topk > 0:
                # Get the top-k probabilities and their corresponding token indices
                top_k_probs, top_k_indices = torch.topk(logits, k=min(topk, logits.size(-1)))
                # Create a tensor of the same size as logits, initialized to negative infinity
                filtered_logits = torch.full_like(logits, -float('inf'))
                # Keep only the top-k logit values
                filtered_logits.scatter_(1, top_k_indices, top_k_probs)
                logits = filtered_logits
            
            # Convert to a probability distribution
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check if the end token has been generated
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token.item())
            x = next_token.reshape(1, 1)
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.info(f"Length of normally generated text: {len(generated_text)}")
        logger.info(f"Number of generated tokens: {len(generated_tokens)}")
        
        return generated_text, generated_tokens
        