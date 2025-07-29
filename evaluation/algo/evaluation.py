
import gc
from hashlib import algorithms_available
import json
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
import torch.nn.functional as F
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import numpy as np


def calculate_ppl(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,text:str):
    """
    Calculates the perplexity (PPL) of a given text.
    Args:
        model:AutoModelForCausalLM: The evaluation model.
        tokenizer:AutoTokenizer: The tokenizer for the evaluation model.
        text:str: The text for which to calculate perplexity.
        
    Returns:
        float: The perplexity score. A lower score indicates more natural text.
    """
    with torch.no_grad():
        # Tokenize the text
        tokenizer_output = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded_text = tokenizer_output["input_ids"][0].to(model.device)
        del tokenizer_output  # Immediately clean up the tokenizer output
        
        # Set the loss function (CrossEntropyLoss)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get model predictions
        model_input = torch.unsqueeze(encoded_text, 0)
        model_output = model(model_input, return_dict=True)
        logits = model_output.logits[0].clone()  # Use clone to avoid reference
        
        # Immediately clean up model output and input
        del model_output, model_input
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate loss and perplexity
        # We predict the next token, so we compare logits[:-1] with encoded_text[1:]
        logits_slice = logits[:-1]
        target_slice = encoded_text[1:]
        loss = criterion(logits_slice, target_slice)
        ppl = torch.exp(loss)  # Perplexity is the exponential of the loss
        # Save the result
        result = ppl.item()
        return result
def calculate_semantic_entropy(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,text:str):
    """
    Calculates the semantic entropy of a given text.
    Args:
        model:AutoModelForCausalLM: The evaluation model.
        tokenizer:AutoTokenizer: The tokenizer for the evaluation model.
        text:str: The text for which to calculate semantic entropy.
        
    Returns:
        float: The semantic entropy score. A higher score indicates greater semantic uncertainty.
    """
    with torch.no_grad():
        # Tokenize the text
        tokenizer_output = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded_text = tokenizer_output["input_ids"][0].to(model.device)
        # Get model predictions
        model_input = torch.unsqueeze(encoded_text, 0)
        model_output = model(model_input, return_dict=True)
        logits = model_output.logits[0].clone()  # Use clone to avoid reference
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Convert logits to a probability distribution
        logits_slice = logits[:-1]  # For all positions except the last one
        probabilities = F.softmax(logits_slice, dim=-1)
        log_probs = torch.log(probabilities + 1e-10)  # Add a small constant to avoid log(0)
        entropy_per_position = -torch.sum(probabilities * log_probs, dim=-1)
        # Calculate the average semantic entropy
        semantic_entropy = torch.mean(entropy_per_position)
        # Save the result
        result = semantic_entropy.item()
        return result


def calculate_rouge1(reference, candidate):
    """
    Calculates the ROUGE-1 score.
    
    ROUGE-1 is an evaluation metric based on unigram matching, used to measure the similarity between a candidate text and a reference text.
    The score ranges from 0 to 1, where 1 indicates a perfect match.
    
    Args:
        reference (str): The reference text (usually the original text).
        candidate (str): The candidate text (usually the generated or watermarked text).
        
    Returns:
        dict: A dictionary containing the ROUGE-1 precision, recall, and F1-score.
    """
    # Ensure that nltk has downloaded the necessary resources
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    # Tokenize the texts
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Calculate unigrams
    reference_unigrams = Counter(reference_tokens)
    candidate_unigrams = Counter(candidate_tokens)
    
    # Calculate common unigrams
    common_unigrams = reference_unigrams & candidate_unigrams
    common_count = sum(common_unigrams.values())
    
    # Calculate precision, recall, and F1-score
    candidate_total = sum(candidate_unigrams.values())
    reference_total = sum(reference_unigrams.values())
    precision = common_count / max(candidate_total, 1)
    recall = common_count / max(reference_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    
    # Save the result
    result = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return result

def calculate_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculates the BLEU score.

    Args:
        reference (str): The reference text (usually the original text).
        candidate (str): The candidate text (usually the generated or watermarked text).
        weights (tuple): The n-gram weights, defaults to (0.25, 0.25, 0.25, 0.25) for BLEU-4.
        
    Returns:
        float: The BLEU score.
    """
    # Ensure that nltk has downloaded the necessary resources
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    # Tokenize the texts
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Use a smoothing function to avoid zero precision
    smoothing = SmoothingFunction().method1
    
    # Calculate the BLEU score
    bleu_score = sentence_bleu(
        [reference_tokens],  # List of reference texts (can have multiple references)
        candidate_tokens,    # Candidate text
        weights=weights,     # n-gram weights
        smoothing_function=smoothing  # Smoothing function
    )
    # Save the result
    result = bleu_score
    return result

def calculate_lexical_diversity(text):
    """
    Calculates the lexical diversity of a given text, without relying on a reference text.

    Args:
        text (str): The text to be evaluated.
        
    Returns:
        dict: A dictionary containing various lexical diversity metrics.
    """
    try:
        # Ensure that the nltk tokenizer is available
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    
    # Filter out non-alphabetic tokens, such as punctuation
    words = [word for word in tokens if word.isalpha()]
    
    if not words:
        # Clean up temporary variables
        del tokens, words
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        return {
            'ttr': 0,
            'rttr': 0,
            'unigram_entropy': 0
        }

    total_tokens = len(words)
    unique_words = set(words)
    unique_tokens = len(unique_words)

    # 1. Calculate TTR (Type-Token Ratio)
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    # 2. Calculate RTTR (Root TTR)
    rttr = unique_tokens / np.sqrt(total_tokens) if total_tokens > 0 else 0

    # 3. Calculate Unigram Entropy
    counts = Counter(words)
    probabilities = [count / total_tokens for count in counts.values()]
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    # Save the result
    result = {
        'ttr': ttr,
        'rttr': rttr,
        'unigram_entropy': entropy
    }
    return result


def parse_conversation(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,conversation_path:str,result_path:str):
    with open(conversation_path, 'r', encoding='utf-8') as file:
        conversation = json.load(file)
    result = {
        "conversation_info":conversation_path,
        "experiment_config":{
            "steganography_algorithm":conversation['sessionInfo']['steganographyAlgorithm'],
            "question_domain":conversation['sessionInfo']['topic'],
            "question_index":conversation['sessionInfo']['questionIndex']
        },
        "average_capacity_metrics":{
            "bits_per_round":conversation['secretMessage']['totalSizeBytes']/len(conversation['rounds']),
            "round_per_bit":len(conversation['rounds'])/conversation['secretMessage']['totalSizeBytes']
        },
        "rounds":[
        ],
        "average_quality_metrics":None
    }
    print(f"Parsing conversation: {conversation_path}")
    print("Experiment Configuration")
    print(f"    Steganography Algorithm: {conversation['sessionInfo']['steganographyAlgorithm']}")
    print(f"    Conversation Domain: {conversation['sessionInfo']['topic']}")
    print(f"    Question Index: {conversation['sessionInfo']['questionIndex']}")
    print("Average Transmission Efficiency Metrics:")
    print(f"    Average bits per round (bits/round): {result['average_capacity_metrics']['bits_per_round']}")
    print(f"    Average rounds per bit (round/bit): {result['average_capacity_metrics']['round_per_bit']}")
    print("Round-by-Round Text Quality Analysis:")
    ppl_list = []
    entropy_list = []
    rouge1_list = []
    bleu_list = []
    lex_div_list = []
    rounds = conversation['rounds']
    for round in rounds:
        stego_text = round['clientTurn']['publicCarrierMessage']
        cover_text = round['clientTurn']['normalMessage']
        ppl = calculate_ppl(model,tokenizer,stego_text)
        entropy = calculate_semantic_entropy(model,tokenizer,stego_text)
        rouge1 = calculate_rouge1(cover_text,stego_text)
        bleu = calculate_bleu(cover_text,stego_text)
        lex_div = calculate_lexical_diversity(stego_text)
        print(f"    Round: {round['roundNumber']}")
        print(f"        Perplexity: {ppl}")
        print(f"        Semantic Entropy: {entropy}")
        print(f"        ROUGE-1 (Precision): {rouge1['precision']}")
        print(f"        ROUGE-1 (Recall): {rouge1['recall']}")
        print(f"        ROUGE-1 (F1): {rouge1['f1']}")
        print(f"        BLEU: {bleu}")
        print(f"        Lexical Diversity (TTR): {lex_div['ttr']}")
        print(f"        Lexical Diversity (RTTR): {lex_div['rttr']}")
        print(f"        Lexical Diversity (Unigram Entropy): {lex_div['unigram_entropy']}")
        result['rounds'].append({
            "round_number":round['roundNumber'],
            "ppl":ppl,
            "entropy":entropy,
            "rouge1_precision":rouge1['precision'],
            "rouge1_recall":rouge1['recall'],
            "rouge1_f1":rouge1['f1'],
            "bleu":bleu,
            "lex_div_ttr":lex_div['ttr'],
            "lex_div_rttr":lex_div['rttr'],
            "lex_div_unigram_entropy":lex_div['unigram_entropy']
        })
        ppl_list.append(ppl)
        entropy_list.append(entropy)
        rouge1_list.append(rouge1)
        bleu_list.append(bleu)
        lex_div_list.append(lex_div)

    print("Average Text Quality Metrics:")
    print(f"    Average Perplexity: {np.mean(ppl_list)}")
    print(f"    Average Semantic Entropy: {np.mean(entropy_list)}")
    print(f"    Average ROUGE-1 (Precision): {np.mean([rouge1['precision'] for rouge1 in rouge1_list])}")
    print(f"    Average ROUGE-1 (Recall): {np.mean([rouge1['recall'] for rouge1 in rouge1_list])}")
    print(f"    Average ROUGE-1 (F1): {np.mean([rouge1['f1'] for rouge1 in rouge1_list])}")
    print(f"    Average BLEU: {np.mean(bleu_list)}")
    print(f"    Average Lexical Diversity (TTR): {np.mean([lex_div['ttr'] for lex_div in lex_div_list])}")
    print(f"    Average Lexical Diversity (RTTR): {np.mean([lex_div['rttr'] for lex_div in lex_div_list])}")
    print(f"    Average Lexical Diversity (Unigram Entropy): {np.mean([lex_div['unigram_entropy'] for lex_div in lex_div_list])}")
    result['average_quality_metrics'] = {
        "ppl":np.mean(ppl_list),
        "entropy":np.mean(entropy_list),
        "rouge1_precision":np.mean([rouge1['precision'] for rouge1 in rouge1_list]),
        "rouge1_recall":np.mean([rouge1['recall'] for rouge1 in rouge1_list]),
        "rouge1_f1":np.mean([rouge1['f1'] for rouge1 in rouge1_list]),
        "bleu":np.mean(bleu_list),
        "lex_div_ttr":np.mean([lex_div['ttr'] for lex_div in lex_div_list]),
        "lex_div_rttr":np.mean([lex_div['rttr'] for lex_div in lex_div_list]),
        "lex_div_unigram_entropy":np.mean([lex_div['unigram_entropy'] for lex_div in lex_div_list])
    }
    result_file_name = f"{result_path}/evaluation_{conversation_path.split('/')[-1]}.json"
    with open(result_file_name,'w',encoding='utf-8') as file:
        json.dump(result,file)
    print(f"Results have been saved to: {result_file_name}")





