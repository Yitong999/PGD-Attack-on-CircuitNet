from torch.utils.data import Dataset,DataLoader,random_split
from typing import Any,Tuple,Optional,Callable
import PIL
import csv
import pathlib
import torch
import torch.nn as nn
from torch.optim import Adam,lr_scheduler
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,RandomCrop,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip
import matplotlib.pyplot as plt
import pickle
import tqdm


from data.attr_dataset import GTSRB
from module.util import get_model

from utils.options import args_parser

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Check if MPS (Apple's Metal Performance Shaders) is available
use_mps = torch.backends.mps.is_available()

# Choose the device based on availability
if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

train_transforms = Compose([
    ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    RandomEqualize(0.4),
    AugMix(),
    RandomHorizontalFlip(0.3),
    RandomVerticalFlip(0.3),
    GaussianBlur((3,3)),
    RandomRotation(30),

    Resize([50,50]),
    ToTensor(),
    
])
validation_transforms =  Compose([
    Resize([28,28]),
    ToTensor(),
    
])

def train_test_split(dataset,train_size):

    train_size = int(train_size * len(dataset))
    test_size = int(len(dataset) - train_size)
    return random_split(dataset,[train_size,test_size])







if __name__ == "__main__":
    args = args_parser()

    _root = args.root
    model_type = args.model
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    num_of_class = args.num_of_class
    lr = args.lr

    
    dataset = GTSRB(root=_root,split="train")
    train_set,validation_set = train_test_split(dataset,train_size=0.8)
    print(f'training size : {len(train_set)}, Validation size : {len(validation_set)}')

    train_set.dataset.transform = train_transforms
    validation_set.dataset.transform = validation_transforms

    
    train_loader = DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
    validation_loader = DataLoader(dataset=validation_set,batch_size=BATCH_SIZE)

    model = get_model(model_type, num_of_class, device).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )

    
    loss = nn.CrossEntropyLoss()
    lr_s = lr_scheduler.LinearLR(optimizer,start_factor=1.0,end_factor=0.5,total_iters=10)
    model.compile(train_data=train_loader,validation_data=validation_loader,epochs=EPOCHS,loss_function=loss,optimizer=optimizer,learning_rate_scheduler=lr_s)

    name = args.save_name
    torch.save(model, f'models/model_{name}.pth') # save model

    transforms = Compose([
        Resize([28,28]),
        ToTensor(),
    
    ])

    testdata = GTSRB(root=_root,split='test',transform=transforms)
    print('testing size :',len(testdata))
    test_dataloader = DataLoader(testdata)

    from sklearn.metrics import accuracy_score

    y_pred = []
    y_true = []
    model = model.eval().to(device)
    with tqdm.tqdm(colour='red',total=len(test_dataloader)) as progress:
    
        with torch.no_grad() : 
            for id,(input,label) in enumerate(iter(test_dataloader)):
                input,label = input.to(device),label.to(device)
                y_true.append(label.item())
                prediction = model.forward(input)
                _,prediction = torch.max(prediction,1)
                y_pred.append(prediction.item())
                
                progress.desc = f'Test Accuracy : {accuracy_score(y_true,y_pred)} '
                progress.update(1)



