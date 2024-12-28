import torch.nn as nn
import torchvision
import torch
from torchvision.models import vgg19, VGG19_Weights
import glob
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torchvision.transforms as transforms
from tqdm import tqdm


def analyze_label_folder(folder_path, model_dict, transform, true_label):
    all_predictions = []
    
    for img_file in os.listdir(folder_path):
        if not img_file.endswith('.JPEG'):
            continue
            
        img_path = os.path.join(folder_path, img_file)
        
        try:
            image = cv2.imread(img_path)
            if image is None or len(image.shape) < 3 or image.shape[2] != 3:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_orig = Image.fromarray(image_rgb)
            im = transform(im_orig).cuda()
            
            for model in model_dict.values():
                pred = get_model_prediction(model, im)
                all_predictions.append(pred)
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    unique_labels = set(all_predictions)
    different_count = sum(1 for pred in all_predictions if pred != true_label)
    
    return len(unique_labels), different_count


def get_model_prediction(model, img):
    with torch.no_grad():
        if img.dim() == 3:
            img = img.unsqueeze(0)
        output = model(img)
        prediction = output.argmax(dim=1).item()
    return prediction

def analyze_predictions(image_path, model_dict, transform, true_label):
    try:
        image = cv2.imread(image_path)
        if image is None or len(image.shape) < 3 or image.shape[2] != 3:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_orig = Image.fromarray(image_rgb)
        im = transform(im_orig).cuda()
        
        predictions = []
        for model in model_dict.values():
            pred = get_model_prediction(model, im)
            predictions.append(pred)
        
        unique_labels = set(predictions)
        different_labels = set([p for p in predictions if p != true_label])
        
        return len(unique_labels), len(different_labels)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def create_visualizations(unique_counts, different_counts):
    plt.figure(figsize=(20, 8))
    x = np.arange(len(unique_counts))
    width = 0.35
    
    plt.bar(x - width/2, unique_counts, width, label='Total Unique Labels', color='skyblue')
    plt.bar(x + width/2, different_counts, width, label='Different Labels', color='lightcoral')
    
    plt.xlabel('Label Index', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Label Prediction Statistics per Class', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    
    plt.xticks(x[::5], [f"{i+1}" for i in range(0, len(unique_counts), 5)], rotation=0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(max(unique_counts), max(different_counts)) * 1.1)
    
    plt.tight_layout()
    plt.savefig('label_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    max_possible = max(max(unique_counts), max(different_counts))
    unique_percentages = [count / max_possible * 100 for count in unique_counts]
    different_percentages = [count / max_possible * 100 for count in different_counts]
    
    plt.figure(figsize=(10, 8))
    boxplot = plt.boxplot([unique_percentages, different_percentages], 
                         labels=['Total Unique\nLabels', 'Different\nLabels'],
                         patch_artist=True)
    
    colors = ['skyblue', 'lightcoral']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.title('Distribution of Label Statistics (Percentage)', fontsize=14, pad=20)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    
    plt.yticks(np.arange(0, 101, 20), [f'{x}%' for x in range(0, 101, 20)])
    
    plt.tight_layout()
    plt.savefig('label_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_dataset_and_visualize(base_path, model_dict, transform):
    unique_counts = []
    different_counts = []
    
    for label_dir in tqdm(sorted(os.listdir(base_path)), desc="Processing labels"):
        if not label_dir.startswith('label_'):
            continue
            
        label_path = os.path.join(base_path, label_dir)
        true_label = int(label_dir.split('_')[1])
        
        unique_count, different_count = analyze_label_folder(
            label_path, model_dict, transform, true_label)
            
        print(f"\nLabel {true_label}:")
        print(f"  Number of unique predicted labels: {unique_count}")
        print(f"  Total number of incorrect predictions: {different_count}")
        
        unique_counts.append(unique_count)
        different_counts.append(different_count)
    
    create_visualizations(unique_counts, different_counts)
    
    with open('label_stats.txt', 'w') as f:
        f.write("=== Label Statistics ===\n\n")
        f.write("Per Label Statistics:\n")
        for i, (unique, different) in enumerate(zip(unique_counts, different_counts), 1):
            f.write(f"Label {i:03d}:\n")
            f.write(f"  Number of unique predicted labels: {unique}\n")
            f.write(f"  Total number of incorrect predictions: {different}\n")
        f.write("\nOverall Statistics:\n")
        f.write(f"Average number of unique labels: {np.mean(unique_counts):.2f}\n")
        f.write(f"Average number of incorrect predictions: {np.mean(different_counts):.2f}\n")
        f.write(f"Max number of unique labels: {np.max(unique_counts)}\n")
        f.write(f"Max number of incorrect predictions: {np.max(different_counts)}\n")

        
if __name__ == "__main__":
    num_models = 3
    Scale = 20
    
    print("Initializing models...")
    net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).cuda()   
    net.eval()
    
    from ModelModify import ModifyModelVGGScale
    model_dict, _ = ModifyModelVGGScale(
        vgg19(weights=VGG19_Weights.IMAGENET1K_V1), 
        num_models, Scale)
    
    for model in model_dict.values():
        model.cuda()
        model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = r"/users/PMIU0184/guoj41/Classification/ILSVRC/dnn_encryption/ILSVRC/organized_dataset/"
    print("Processing dataset and generating visualizations...")
    process_dataset_and_visualize(dataset_path, model_dict, transform)
    
    print("Analysis completed! Results saved to prediction_analysis.png and prediction_stats.txt")
