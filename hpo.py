#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import vgg19
import argparse
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    model.eval() 
    
    losses = []
    corrects=0  
    
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)
        outputs=model(data)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        losses.append(loss.item())             # calculate running loss
        corrects += torch.sum(preds == labels.data)     # calculate running corrects

    avg_loss = np.mean(losses)
    avg_acc = np.true_divide(corrects.double().cpu(), len(test_loader.dataset))
    
    logger.info('Testing Loss: {:.4f}, Accuracy: {:.4f}'.format(avg_loss, avg_acc)) # print the avg loss and accuracy values
    
def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):

    best_loss = np.Inf #initialize best loss to infinity
    dataset_dict={'train':train_loader, 'valid':valid_loader}
    
    for epoch in range(1, epochs+1):
        logger.info(f"Epoch:{epoch}---")
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            losses = []
            corrects = 0
            
            for data, labels in dataset_dict[phase]:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                
                losses.append(loss.item())
                corrects += torch.sum(preds == labels.data)

            avg_loss = np.mean(losses)
            avg_acc = np.true_divide(corrects.cpu(), len(dataset_dict[phase].dataset))
            
            if phase=='valid':
                if avg_loss < best_loss:
                    best_loss = avg_loss
            
            logger.info('{} Loss: {:.4f}, Accuracy: {:.4f}, Best Loss: {:.4f}'.format(phase.capitalize(),
                                                                                 avg_loss,
                                                                                 avg_acc,
                                                                                 best_loss))
    return model
    
def net(num_classes):
    vgg = models.vgg19(pretrained=True)
    in_features = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(in_features, num_classes)
    return vgg

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path = os.path.join(data, 'train') # dogImages/train
    test_data_path = os.path.join(data, 'test') # dogImages/test
    valid_data_path = os.path.join(data, 'valid') # dogImages/valid
    
    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    ])

    train_loader = DataLoader(
        ImageFolder(train_data_path, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        ImageFolder(test_data_path, transform=transform),
        batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        ImageFolder(valid_data_path, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, valid_loader

def save_model(model, model_dir):
    logger.info("Saving the model...")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main(args):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    logger.info(f"Running on Device {device}")
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Path: {args.data_dir}')
    
    model=net(args.num_classes)
    model = model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)

    
    logger.info("Training model...")
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs)
    
    logger.info("Testing the model...")
    test(model, test_loader, loss_criterion, device)
    
    # Saving the model
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="epochs (default: 2)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=133,
        metavar="N",
        help="number of classes (default: 133)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    # parser.add_argument(
    #     "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    # )

    # Container environment
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
