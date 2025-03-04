import torch
import torch.nn.functional as F
import operator
import math

def softmax_entropy(x):
    """Calculate entropy of softmax distribution"""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_entropy(loss, num_classes):
    """Normalize entropy to [0, 1] range"""
    max_entropy = math.log2(num_classes)
    return float(loss / max_entropy)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """
    Update cache with new features and loss values, maintaining max cache capacity.
    
    Args:
        cache (dict): Cache dictionary, keys are predicted classes, values are lists of feature data and loss values.
        pred (int): Predicted class for the new feature data.
        features_loss (list): List containing feature data and loss value, possibly also probability map.
        shot_capacity (int): Maximum cache capacity per class.
        include_prob_map (bool): Whether to include probability map in cache data, default is False.
    """
    with torch.no_grad():
        # Determine what data items to cache based on include_prob_map
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        
        if pred in cache:
            # If cache for current class is not full, add new item directly
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            # If cache is full and new item has lower loss than the highest loss in cache, replace that item
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            # Sort cache by loss value to maintain low-to-high order
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # If cache for current class is empty, initialize it and add new item
            cache[pred] = [item]

def compute_cache_logits(box_features, cache, alpha, beta, num_classes, neg_mask_thresholds=None):
    """
    Compute logits using positive/negative cache.
    
    Args:
        box_features (torch.Tensor): Box features.
        cache (dict): Cache dictionary, keys are predicted classes, values are lists of feature data and loss values.
        alpha (float): Weight factor.
        beta (float): Weight factor.
        num_classes (int): Number of classes.
        neg_mask_thresholds (tuple, optional): Thresholds range for negative cache. Default is None.
        
    Returns:
        torch.Tensor: Computed logits.
    """
    with torch.no_grad():
        if not cache:  # If cache is empty, return zero tensor
            return torch.zeros((box_features.size(0), num_classes), device=box_features.device)
            
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        # Move all data to the same device as box_features
        device = box_features.device
        cache_keys = torch.cat(cache_keys, dim=0).to(device)
        
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0).to(device)
            cache_values = ((cache_values > neg_mask_thresholds[0]) & 
                        (cache_values < neg_mask_thresholds[1])).to(torch.float32)
        else:
            cache_values = F.one_hot(
                torch.tensor(cache_values, dtype=torch.int64, device=device), 
                num_classes=num_classes
            ).float()  # 添加.float()转换为浮点型
            
        # Calculate affinity between box features and cache keys
        affinity = box_features @ cache_keys.t()
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits