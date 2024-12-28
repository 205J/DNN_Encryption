import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from torchvision.utils import save_image
from tqdm import tqdm
from torchplus.utils import Init, ClassificationAccuracy
import copy
import random
import numpy as np


if __name__ == "__main__":
    batch_size = 8
    train_epoches = 50
    log_epoch = 2
    class_num = 40
    root_dir = r"C:\Users\ADMIN\lpl\MIA\logs"
    dataset_dir = r"C:\Users\ADMIN\lpl\at&t face database"
    h = 112
    w = 92

    init = Init(
        seed=9970,
        log_root_dir=root_dir,
        sep=True,
        backup_filename=__file__,
        tensorboard=True,
        comment=f"main AT and T face",
    )
    output_device = init.get_device()
    writer = init.get_writer()
    log_dir, model_dir = init.get_log_dir()
    data_workers = 2

    transform = Compose([Grayscale(num_output_channels=1), ToTensor()])

    ds = ImageFolder(dataset_dir, transform=transform)
    ds_len = len(ds)
    train_ds, test_ds = random_split(ds, [ds_len * 7 // 10, ds_len - ds_len * 7 // 10])

    train_ds_len = len(train_ds)
    test_ds_len = len(test_ds)

    print(train_ds_len)
    print(test_ds_len)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        drop_last=False,
        pin_memory=True,
    )

    class Net(nn.Module):
        def __init__(self, input_features, output_features):
            super(Net, self).__init__()
            self.input_features = input_features
            self.output_features = output_features
            self.regression = nn.Linear(
                in_features=self.input_features, out_features=self.output_features
            )

        def forward(self, x):
            x = self.regression(x)
            return x

    class MLP(nn.Module):
        def __init__(self, input_features, output_features):
            super(MLP, self).__init__()
            self.input_features = input_features
            self.middle_features = 3000
            self.output_features = output_features
            self.fc = nn.Linear(
                in_features=self.input_features, out_features=self.middle_features
            )
            self.regression = nn.Linear(
                in_features=self.middle_features, out_features=self.output_features
            )

        def forward(self, x):
            x = self.fc(x)
            x = self.regression(x)
            return x

    def ModifyModelScale(net, Scale, split_num=1):
        if isinstance(net, MLP):
            TTB = net.regression.weight
            BiaB = net.regression.bias
        else:
            TTB = net.regression.weight
            BiaB = net.regression.bias

        r, c = TTB.shape
        
        BSave = np.zeros((1, r))
        for i in range(r):
            BSave[0][i] = BiaB[i].clone().cpu().numpy()
        
        BSaveB = BSave.copy()
        MaxBias = BSaveB.max()
        for i in range(r):
            Flag = 1
            if BSaveB[0][i] < 0:
                Flag = -1
            BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
        
        WSave = TTB.clone().cpu().numpy()
        
        WSaveB = WSave.copy()
        WeightMax = WSaveB.max()
        for i in range(r):
            for j in range(c):
                Flag = 1
                if WSaveB[i][j] < 0:
                    Flag = -1
                WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag
        
        OutL = int(r * 2)
        if isinstance(net, MLP):
            net.regression = nn.Linear(in_features=c, out_features=OutL, bias=True)
        else:
            net.regression = nn.Linear(in_features=c, out_features=OutL, bias=True)
        NewWeight = net.regression.weight
        NewBias = net.regression.bias

        order_mapping = []
        with torch.no_grad():
            for i in range(r):
                if random.random() < 0.5:
                    NewWeight[2*i] = torch.tensor(WSave[i])
                    NewBias[2*i] = torch.tensor(BSave[0][i])
                    NewWeight[2*i + 1] = torch.tensor(WSaveB[i])
                    NewBias[2*i + 1] = torch.tensor(BSaveB[0][i])
                    order_mapping.extend([('original', i), ('fake', i)])
                else:
                    NewWeight[2*i] = torch.tensor(WSaveB[i])
                    NewBias[2*i] = torch.tensor(BSaveB[0][i])
                    NewWeight[2*i + 1] = torch.tensor(WSave[i])
                    NewBias[2*i + 1] = torch.tensor(BSave[0][i])
                    order_mapping.extend([('fake', i), ('original', i)])

        net.order_mapping = order_mapping

        return net

    def DecodeOutput(net, output):
        decoded_output = torch.zeros(class_num, device=output.device)
        for i, (type, original_index) in enumerate(net.order_mapping):
            if type == 'original':
                decoded_output[original_index] = output[i]
        return decoded_output

    mynet = Net(h * w, class_num).to(output_device).train(True)
    optimizer = optim.SGD(mynet.parameters(), lr=0.01)

    for epoch_id in tqdm(range(1, train_epoches + 1), desc="Total Epoch"):
        iters = tqdm(train_dl, desc=f"epoch {epoch_id}")
        for i, (im, label) in enumerate(iters):
            im = im.to(output_device)
            label = label.to(output_device)
            bs, c, h, w = im.shape
            optimizer.zero_grad()
            im_flatten = im.reshape([bs, -1])
            out = mynet.forward(im_flatten)
            ce = nn.CrossEntropyLoss()(out, label)
            loss = ce
            loss.backward()
            optimizer.step()

        if epoch_id % log_epoch == 0:
            train_ca = ClassificationAccuracy(class_num)
            after_softmax = F.softmax(out, dim=-1)
            predict = torch.argmax(after_softmax, dim=-1)
            train_ca.accumulate(label=label, predict=predict)
            acc_train = train_ca.get()
            writer.add_scalar("loss", loss, epoch_id)
            writer.add_scalar("acc_training", acc_train, epoch_id)
            with open(os.path.join(model_dir, f"mynet_{epoch_id}.pkl"), "wb") as f:
                torch.save(mynet.state_dict(), f)

            with torch.no_grad():
                mynet.eval()
                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(train_dl, desc="testing train")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    im_flatten = im.reshape([bs, -1])
                    out = mynet.forward(im_flatten)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("train loss", celossavg, epoch_id)
                writer.add_scalar("acc_train", acc_test, epoch_id)

                r = 0
                celoss = 0
                test_ca = ClassificationAccuracy(class_num)
                for i, (im, label) in enumerate(tqdm(test_dl, desc="testing test")):
                    r += 1
                    im = im.to(output_device)
                    label = label.to(output_device)
                    bs, c, h, w = im.shape
                    im_flatten = im.reshape([bs, -1])
                    out = mynet.forward(im_flatten)
                    ce = nn.CrossEntropyLoss()(out, label)
                    after_softmax = F.softmax(out, dim=-1)
                    predict = torch.argmax(after_softmax, dim=-1)
                    test_ca.accumulate(label=label, predict=predict)
                    celoss += ce

                celossavg = celoss / r
                acc_test = test_ca.get()
                writer.add_scalar("test loss", celossavg, epoch_id)
                writer.add_scalar("acc_test", acc_test, epoch_id)

    print("Applying defense mechanism...")
    Scale = 20  # You can adjust this value
    mynet = ModifyModelScale(mynet, Scale)

    def evaluate_model(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(output_device), labels.to(output_device)
                outputs = model(images.reshape(images.shape[0], -1))
                decoded_outputs = DecodeOutput(model, outputs)
                _, predicted = torch.max(decoded_outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    train_accuracy = evaluate_model(mynet, train_dl)
    test_accuracy = evaluate_model(mynet, test_dl)
    print(f"Train Accuracy after defense: {train_accuracy:.4f}")
    print(f"Test Accuracy after defense: {test_accuracy:.4f}")

    # Save the final model with defense mechanism
    torch.save({
        'model_state_dict': mynet.state_dict(),
        'order_mapping': mynet.order_mapping
    }, os.path.join(model_dir, "mynet_final_with_defense.pkl"))

    writer.close()