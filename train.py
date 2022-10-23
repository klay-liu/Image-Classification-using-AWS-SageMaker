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

#TODO: Import dependencies for Debugging andd Profiling
    
def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train progress: [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    
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
    pass

def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    # logger.info(f"{os.listdir(training_dir)}")
    # train_path = os.path.join(training_dir, 'train')
    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    ])

    train_loader = DataLoader(
        ImageFolder(training_dir, transform=transform),
        batch_size=batch_size, shuffle=True)
    return train_loader
    
def _get_test_data_loader(batch_size, testing_dir):
    logger.info("Get test data loader")
    # logger.info(f"{os.listdir(testing_dir)}")
    # test_path = os.path.join(training_dir, 'valid')
   
    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    ])

    test_loader = DataLoader(
        ImageFolder(testing_dir, transform=transform),
        batch_size=batch_size, shuffle=True)
    return test_loader

def model_fn(model_dir):
    model = net(num_classes=133)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model=net(args.num_classes)
    logger.info(f'model: {model}')
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = _get_train_data_loader(args.train_batch_size, args.data_dir)
    
    train(model, train_loader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = _get_test_data_loader(args.test_batch_size, args.testing_data_dir)
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        metavar="N",
        help="number of classes (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--testing_data_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    
    args=parser.parse_args()
    
    main(args)
