import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch

def load_pkl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def prepare_data(data, predefined_labels=None):
    if isinstance(data, dict):
        labels = list(data.keys())
        embeddings = list(data.values())
    elif isinstance(data, torch.Tensor):
        embeddings = data.cpu().numpy()
        if predefined_labels:
            if len(embeddings) != len(predefined_labels):
                raise ValueError("Number of embeddings does not match number of predefined labels.")
            labels = predefined_labels
        else:
            labels = [f'Class {i}' for i in range(len(embeddings))]
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        embeddings = np.array(data)
        labels = [f'Class {i}' for i in range(len(embeddings))]
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    return labels, embeddings

def reduce_dimensionality(embeddings, method='tsne', n_components=2, random_state=42, perplexity=5):
    if method == 'tsne':
        if len(embeddings) <= perplexity:
            perplexity = max(1, len(embeddings) // 2)
            print(f"Adjusted perplexity to {perplexity} due to small number of samples.")
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, n_iter=1000)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    reduced = reducer.fit_transform(embeddings)
    return reduced

def plot_embeddings(reduced_embeddings, labels, title='Text Embeddings Visualization', figsize=(12, 8)):
    plt.figure(figsize=figsize)
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[idxs, 0], reduced_embeddings[idxs, 1], 
                    label=label, color=label_to_color[label], s=100, alpha=0.8)
    
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    pkl_file_path = r'F:\googledownload\WSOVOD\voc_text_embedding_single_prompt.pkl'
    
    try:
        data = load_pkl(pkl_file_path)
        print(f"Loaded data with type: {type(data)}")
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        return
    
    # Define VOC classes
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    
    try:
        labels, embeddings = prepare_data(data, predefined_labels=voc_classes)
        embeddings = np.array(embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
    try:
        reduced = reduce_dimensionality(embeddings, method='tsne', n_components=2, perplexity=5)
        print("Dimensionality reduction completed.")
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        return
    
    try:
        plot_embeddings(reduced, labels)
        print("Visualization completed.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        return

if __name__ == "__main__":
    main()
