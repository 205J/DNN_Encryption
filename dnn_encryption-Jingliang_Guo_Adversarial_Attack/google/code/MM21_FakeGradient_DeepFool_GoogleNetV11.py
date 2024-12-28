'''
Modified version: Simplified evaluation focusing on model accuracy and defense effectiveness
'''

import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
import os
import csv
from ModelModify import ModifyModelScale
from DeepFoolC import deepfoolC
from DeepFoolB import deepfoolB
import cv2
from torchvision.models import googlenet, GoogLeNet_Weights
import glob
def get_fc_output(model, x):

    model.eval()
    with torch.no_grad():
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = model(x)
        return x

def decrypt_model_outputs(output_dict, order_mapping, num_models):
    """
    Decrypt the outputs from multiple models
    """
    processed_dict = {}
    for key, value in output_dict.items():
        if isinstance(value, np.ndarray):
            processed_dict[key] = torch.from_numpy(value).cuda()
        else:
            processed_dict[key] = value
    
    device = next(iter(processed_dict.values())).device
    batch_size = next(iter(processed_dict.values())).shape[0]
    original_vocab_size = sum(1 for type_, _ in order_mapping if type_ == 'original')
    
    group_outputs = []
    for group_idx in range(2):
        group_tensors = []
        for model_idx in range(num_models):
            key = f"{group_idx}_{model_idx}"
            if key in processed_dict:
                group_tensors.append(processed_dict[key])
        
        group_concat = torch.cat(group_tensors, dim=1)
        decrypted = torch.zeros(batch_size, original_vocab_size, device=device)
        
        for current_pos, (type_, orig_idx) in enumerate(order_mapping):
            if type_ == 'original' and orig_idx < original_vocab_size:
                decrypted[:, orig_idx] = group_concat[:, current_pos]
        
        group_outputs.append(decrypted)
    
    return group_outputs[0] + group_outputs[1]

def get_model_prediction(model, img):
    """
    Get model's prediction for an image
    """
    with torch.no_grad():
        output = get_fc_output(model, img)
        prediction = torch.nn.functional.softmax(output, dim=1).argmax().item()
    return prediction

def process_and_attack_models(image_path, net, model_dict, transform, order_mapping, num_models):
    try:
        image = cv2.imread(image_path)
        if image is None or len(image.shape) < 3 or image.shape[2] != 3:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_orig = Image.fromarray(image_rgb)
        im = transform(im_orig).cuda()
        original_pred = get_model_prediction(net, im)
        rB, loop_iB, label_origB, label_pertB, pert_imageB, gradient = deepfoolB(im, net)
        encrypted_outputs = {}
        encrypted_attack_outputs = {}
        for key, model in model_dict.items():
            fc_output = get_fc_output(model, im)
            encrypted_outputs[key] = fc_output
            r, loop_i, label_orig, label_pert, Originallabel, Protected, pert_image, _ = deepfoolC(
                im, model)
            if isinstance(pert_image, np.ndarray):
                pert_image = torch.from_numpy(pert_image).cuda()
            attack_output = get_fc_output(model, pert_image)
            encrypted_attack_outputs[key] = attack_output
        decrypted_output = decrypt_model_outputs(encrypted_outputs, order_mapping, num_models)
        decrypted_attack_output = decrypt_model_outputs(encrypted_attack_outputs, order_mapping, num_models)
        with torch.no_grad():
            decrypted_pred = torch.nn.functional.softmax(decrypted_output, dim=1).argmax().item()
            decrypted_attack_pred = torch.nn.functional.softmax(decrypted_attack_output, dim=1).argmax().item()
        return original_pred, label_pertB, decrypted_pred, decrypted_attack_pred
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def write_statistics(writer, file_obj, CountTotal, original_integrity, decrypted_integrity, 
                    decrypted_attack_integrity, is_final=False):
    """
    Write statistics to CSV file
    """
    status = "Final Results" if is_final else f"Progress ({CountTotal} images)"
    stats = [
        [status],
        ["Model Integrity Metrics:"],
        ["1. Original Model After Attack", f"{original_integrity/CountTotal*100:.2f}%"],
        ["2. Encrypted-Decrypted Model", f"{decrypted_integrity/CountTotal*100:.2f}%"],
        ["3. Encrypted-Decrypted Model After Attack", f"{decrypted_attack_integrity/CountTotal*100:.2f}%"],
        ["Total Processed Images", str(CountTotal)],
        [""]
    ]
    
    for row in stats:
        writer.writerow(row)
    file_obj.flush()


if __name__ == "__main__":
    # Setup models
    num_models = 3
    Scale = 20
    
    print("Initializing models...")
    net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).cuda()
    net.eval()
    
    model_dict, order_mapping =ModifyModelScale(
        googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1), 
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
    
    # Setup statistics
    CSVfilenameTime = 'google_Model_Integrity_Results1.csv'
    print(f"Results will be saved to: {CSVfilenameTime}")
    
    with open(CSVfilenameTime, 'w', newline='') as fileobjT:
        writerT = csv.writer(fileobjT)
        
        CountTotal = 0
        original_integrity = 0 
        decrypted_integrity = 0
        decrypted_attack_integrity = 0
        
        Folder = r"/users/PMIU0184/guoj41/Classification/ILSVRC/Data/DET/train/folder1"
        
        
        file_pattern = os.path.join(Folder, "*.*")
        image_files = glob.glob(file_pattern)
        image_files.sort()

        total_files = len(image_files)
        print(f"Found {total_files} files to process")
        writerT.writerow(["Processing Start"])
        writerT.writerow(["Total files to process:", str(total_files)])
        writerT.writerow([])
        
        for image_path in image_files:
            result = process_and_attack_models(
                image_path=image_path,
                net=net,
                model_dict=model_dict,
                transform=transform,
                order_mapping=order_mapping,
                num_models=num_models
            )
            
            if result is None:
                continue

            original_pred, attacked_pred, decrypted_pred, decrypted_attack_pred = result
            CountTotal += 1
            
            # Update integrity metrics
            if original_pred == attacked_pred:
                original_integrity += 1
            if original_pred == decrypted_pred:
                decrypted_integrity += 1
            if original_pred == decrypted_attack_pred:
                decrypted_attack_integrity += 1
            
            # Print progress every 100 images
            if CountTotal % 100 == 0:
                print(f"\nProcessed images: {CountTotal}")
                print("Model Integrity Metrics:")
                print(f"1. Original Model After Attack: {original_integrity/CountTotal*100:.2f}%")
                print(f"2. Encrypted-Decrypted Model: {decrypted_integrity/CountTotal*100:.2f}%")
                print(f"3. Encrypted-Decrypted Model After Attack: {decrypted_attack_integrity/CountTotal*100:.2f}%")
                
                write_statistics(writerT, fileobjT, CountTotal, original_integrity, 
                               decrypted_integrity, decrypted_attack_integrity)
        
        if CountTotal > 0:
            write_statistics(writerT, fileobjT, CountTotal, original_integrity,
                           decrypted_integrity, decrypted_attack_integrity, is_final=True)

    print("Processing completed!")
