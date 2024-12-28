import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models

from PIL import Image
from HeatMapShow import ShowHeatMap
import AnalysisWeight as AW
import copy



def ModifyModelVGGScale(net1, num_models=2, scale=1):
    original_weight = net1.classifier[6].weight  # [1000, 4096]
    original_bias = net1.classifier[6].bias      # [1000]
    vocab_size, hidden_size = original_weight.shape
    
    encrypted_weight, encrypted_bias, order_mapping = encrypt_weights_and_bias(
        original_weight, original_bias, num_models, scale
    )
    
    random_ratios = torch.rand_like(encrypted_weight)
    
    split_weights = []
    split_biases = []
    
    current_ratio = random_ratios
    first_weight = encrypted_weight * current_ratio
    first_bias = encrypted_bias * current_ratio.mean(dim=1)
    split_weights.append(first_weight)
    split_biases.append(first_bias)
    
    # Second part (remaining)
    second_weight = encrypted_weight * (1 - current_ratio)
    second_bias = encrypted_bias * (1 - current_ratio.mean(dim=1))
    split_weights.append(second_weight)
    split_biases.append(second_bias)
    
    # Size-splitting step - Uniformly split weight matrix rows
    final_weights = []
    final_biases = []
    
    expanded_vocab_size = vocab_size * num_models  # 3000
    rows_per_split = expanded_vocab_size // num_models  # 1000
    
    for split_weight, split_bias in zip(split_weights, split_biases):
        # Split weights and biases by rows
        for i in range(num_models):
            start_row = i * rows_per_split
            end_row = (i + 1) * rows_per_split
            
            weight_part = split_weight[start_row:end_row, :]  # [1000, 4096]
            bias_part = split_bias[start_row:end_row]         # [1000]
            
            final_weights.append(weight_part)
            final_biases.append(bias_part)
    
    
    model_dict = {}
    total_models = 2 * num_models
    
    for i in range(total_models):
        group_idx = i // num_models
        in_group_idx = i % num_models
        model_key = f"{group_idx}_{in_group_idx}"
        
        model_copy = copy.deepcopy(net1)
        with torch.no_grad():
            in_features = hidden_size  # Keep as 4096
            out_features = rows_per_split  # Set to 1000
            
            model_copy.classifier[6] = nn.Linear(in_features, out_features)
            model_copy.classifier[6].weight.copy_(final_weights[i])
            model_copy.classifier[6].bias.copy_(final_biases[i])
        
        model_dict[model_key] = model_copy
    
    return model_dict, order_mapping

def encrypt_weights_and_bias(original_weight, original_bias, num_models=2, scale=1):
    device = original_weight.device
    original_dtype = original_weight.dtype
    vocab_size, hidden_size = original_weight.shape
    
    with torch.no_grad():
        # Step 1: Create expanded weight matrix and bias vector
        expanded_weight = torch.zeros(vocab_size * num_models, hidden_size, dtype=original_dtype, device=device)
        expanded_weight[:vocab_size] = original_weight
        weight_max = original_weight.abs().max()
        
        expanded_bias = torch.zeros(vocab_size * num_models, dtype=original_dtype, device=device)
        expanded_bias[:vocab_size] = original_bias
        bias_max = original_bias.abs().max()
        
        # Step 2: Generate fake weights and biases
        for i in range(1, num_models):
            start = i * vocab_size
            end = (i + 1) * vocab_size
            
            random_indices = torch.randint(0, vocab_size, (vocab_size, 7), device=device)
            
            # Process weights
            weight_avg_values = original_weight[random_indices].mean(dim=1)
            weight_flag = torch.where(weight_avg_values >= 0, 1.0, -1.0)
            fake_weights = (weight_max - weight_avg_values.abs()) * scale * weight_flag
            expanded_weight[start:end] = fake_weights.expand(vocab_size, hidden_size)
            
            # Process biases
            bias_avg_values = original_bias[random_indices[:, 0]]
            bias_flag = torch.where(bias_avg_values >= 0, 1.0, -1.0)
            fake_bias = (bias_max - bias_avg_values.abs()) * scale * bias_flag
            expanded_bias[start:end] = fake_bias
        
        # Step 3: Shuffle weights and biases
        shuffled_indices = torch.randperm(vocab_size * num_models, device=device)
        shuffled_weight = expanded_weight[shuffled_indices]
        shuffled_bias = expanded_bias[shuffled_indices]
        
        # Create order mapping
        order_mapping = [(('original' if i < vocab_size else 'fake'), i % vocab_size) 
                        for i in range(vocab_size * num_models)]
        order_mapping = [order_mapping[i] for i in shuffled_indices.cpu().numpy()]
    
    return shuffled_weight, shuffled_bias, order_mapping


def ModifyModelScale(net1, num_models=2, scale=1):
    original_weight = net1.fc.weight  # [1000, 4096]
    original_bias = net1.fc.bias      # [1000]
    vocab_size, hidden_size = original_weight.shape
    
    encrypted_weight, encrypted_bias, order_mapping = encrypt_weights_and_bias(
        original_weight, original_bias, num_models, scale
    )
    
    random_ratios = torch.rand_like(encrypted_weight)
    
    split_weights = []
    split_biases = []
    
    current_ratio = random_ratios
    first_weight = encrypted_weight * current_ratio
    first_bias = encrypted_bias * current_ratio.mean(dim=1)
    split_weights.append(first_weight)
    split_biases.append(first_bias)
    
    # Second part (remaining)
    second_weight = encrypted_weight * (1 - current_ratio)
    second_bias = encrypted_bias * (1 - current_ratio.mean(dim=1))
    split_weights.append(second_weight)
    split_biases.append(second_bias)
    
    # Size-splitting step - Uniformly split weight matrix rows
    final_weights = []
    final_biases = []
    
    expanded_vocab_size = vocab_size * num_models  # 3000
    rows_per_split = expanded_vocab_size // num_models  # 1000
    
    for split_weight, split_bias in zip(split_weights, split_biases):
        # Split weights and biases by rows
        for i in range(num_models):
            start_row = i * rows_per_split
            end_row = (i + 1) * rows_per_split
            
            weight_part = split_weight[start_row:end_row, :]  # [1000, 4096]
            bias_part = split_bias[start_row:end_row]         # [1000]
            
            final_weights.append(weight_part)
            final_biases.append(bias_part)
    
    
    model_dict = {}
    total_models = 2 * num_models
    
    for i in range(total_models):
        group_idx = i // num_models
        in_group_idx = i % num_models
        model_key = f"{group_idx}_{in_group_idx}"
        
        model_copy = copy.deepcopy(net1)
        with torch.no_grad():
            in_features = hidden_size  # Keep as 4096
            out_features = rows_per_split  # Set to 1000
            model_copy.fc = nn.Linear(in_features, out_features)
            model_copy.fc.weight.copy_(final_weights[i])
            model_copy.fc.bias.copy_(final_biases[i])
        model_dict[model_key] = model_copy
    
    return model_dict, order_mapping
