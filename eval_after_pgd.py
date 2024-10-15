
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

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

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


def pgd_attack_adv(device, model_b, model_d, images, labels, eps=0.04, alpha=4/255, lmd = 2, iters=40) :
    images = images.to(device)
    labels = labels.to(device)

    loss = nn.CrossEntropyLoss(reduction='none')
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs_b = model_b(images)
        outputs_d = model_d(images)
        # outputs1 = model(images)
        # output2
        model_b.zero_grad()
        model_d.zero_grad()

        cost_b = loss(outputs_b, labels).to(device)
        

        cost_d = loss(outputs_d, labels).to(device)

        cost = (cost_b - lmd * cost_d).mean()
        
        
        cost.backward()
        # cost 1
        # cost 2
        # cost1-lambda*cost2
        

        adv_images = images + alpha*images.grad.sign()
        
        
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    
    mode = 1
    
    if mode == 0:
        print('###################')
        print('label: ', labels)
        print('ori predicted by biased model: ', torch.argmax(model_b(ori_images), dim=1))
        print('ori predicted by debiased model: ', torch.argmax(model_d(ori_images), dim=1))
        print('adv predicted by biased model: ', torch.argmax(model_b(images), dim=1))
        print('adv predicted by debiased model: ', torch.argmax(model_d(images), dim=1))
    else:
        return images
    
        
    return images


if __name__ == "__main__":
    args = args_parser()

    _root = args.root
    writer = SummaryWriter('runs/hack_on_model_CNN')

    transforms = Compose([
        Resize([28,28]),
        ToTensor(),
    
    ])

    testdata = GTSRB(root=_root,split='test',transform=transforms)
    print('testing size :',len(testdata))
    test_dataloader = DataLoader(testdata)

    from sklearn.metrics import accuracy_score

    y_pred_1_ori = []
    y_pred_1 = []
    y_true_1 = []

    y_pred_2_ori = []
    y_pred_2 = []
    y_true_2 = []

    y_pred_3_ori = []
    y_pred_3 = []
    y_true_3 = []

    y_pred_4_ori = []
    y_pred_4 = []
    y_true_4 = []

    
    model_1 = torch.load('models/model_CNN.pth')
    model_2 = torch.load('models/model_MLP.pth')
    model_3 = torch.load('models/model_MLP.pth')
    model_4 = torch.load('models/model_CNN.pth')

    model_1 = model_1.eval().to(device)
    model_2 = model_2.eval().to(device)
    model_3 = model_3.eval().to(device)
    model_4 = model_4.eval().to(device)

    accuracy_scores_ori_model_1 = []
    accuracy_scores_adv_model_1 = []
    accuracy_scores_ori_model_2 = []
    accuracy_scores_adv_model_2 = []
    accuracy_scores_ori_model_3 = []
    accuracy_scores_adv_model_3 = []
    accuracy_scores_ori_model_4 = []
    accuracy_scores_adv_model_4 = []

    with tqdm.tqdm(colour='red',total=len(test_dataloader)) as progress:
        # with torch.no_grad() : 
        i = 0
        for id,(input,label) in enumerate(iter(test_dataloader)):
            i += 1
            if i ==  500:
                break
            input,label = input.to(device),label.to(device)


            adv_images = pgd_attack_adv(device, model_2, model_1, input, label)

            writer.add_image('ori_images', input[0], id)
            writer.add_image('adv_images', adv_images[0], id)

            #evaluate model_1
            y_true_1.append(label.item())

            prediction_1 = model_1.forward(adv_images)
            _,prediction_1 = torch.max(prediction_1,1)
            y_pred_1.append(prediction_1.item())

            prediction_ori_1 = model_1.forward(input)
            _,prediction_ori_1 = torch.max(prediction_ori_1, 1)
            y_pred_1_ori.append(prediction_ori_1.item())
            

            #evaluate model_2
            y_true_2.append(label.item())

            prediction_2 = model_2.forward(adv_images)
            _,prediction_2 = torch.max(prediction_2,1)
            y_pred_2.append(prediction_2.item())

            prediction_ori_2 = model_2.forward(input)
            _,prediction_ori_2 = torch.max(prediction_ori_2, 1)
            y_pred_2_ori.append(prediction_ori_2.item())

            #evaluate model_3
            y_true_3.append(label.item())

            prediction_3 = model_3.forward(adv_images)
            _,prediction_3 = torch.max(prediction_3,1)
            y_pred_3.append(prediction_3.item())

            prediction_ori_3 = model_3.forward(input)
            _,prediction_ori_3 = torch.max(prediction_ori_3, 1)
            y_pred_3_ori.append(prediction_ori_3.item())

            #evaluate model_4
            y_true_4.append(label.item())

            prediction_4 = model_4.forward(adv_images)
            _,prediction_4 = torch.max(prediction_4,1)
            y_pred_4.append(prediction_4.item())

            prediction_ori_4 = model_4.forward(input)
            _,prediction_ori_4 = torch.max(prediction_ori_4, 1)
            y_pred_4_ori.append(prediction_ori_4.item())


            # progress.desc = f'Test Accuracy for model 1 on adv images: {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1))}, on ori images {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1_ori))}; for model 2:  {"{:.3f}".format(accuracy_score(y_true_2,y_pred_2))}, on ori images {"{:.3f}".format(accuracy_score(y_true_2,y_pred_2_ori))}'
            
            progress.desc = f'Test Accuracy for model 1 on adv images: {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1))}, on ori images {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1_ori))}; for model 3:  {"{:.3f}".format(accuracy_score(y_true_3,y_pred_3))}, on ori images {"{:.3f}".format(accuracy_score(y_true_3,y_pred_3_ori))}'
            # progress.desc = f'Test Accuracy for model 4 on adv images: {"{:.3f}".format(accuracy_score(y_true_4,y_pred_4))}, on ori images {"{:.3f}".format(accuracy_score(y_true_1,y_pred_1_ori))}; for model 2:  {"{:.3f}".format(accuracy_score(y_true_2,y_pred_2))}, on ori images {"{:.3f}".format(accuracy_score(y_true_2,y_pred_2_ori))}'

            accuracy_scores_ori_model_1.append(float("{:.3f}".format(accuracy_score(y_true_1,y_pred_1_ori)))) # CNN_20
            accuracy_scores_adv_model_1.append(float("{:.3f}".format(accuracy_score(y_true_1,y_pred_1))))

            accuracy_scores_ori_model_2.append(float("{:.3f}".format(accuracy_score(y_true_2,y_pred_2_ori)))) # MLP_50
            accuracy_scores_adv_model_2.append(float("{:.3f}".format(accuracy_score(y_true_2,y_pred_2))))

            accuracy_scores_ori_model_3.append(float("{:.3f}".format(accuracy_score(y_true_3,y_pred_3_ori)))) # MLP_100
            accuracy_scores_adv_model_3.append(float("{:.3f}".format(accuracy_score(y_true_3,y_pred_3))))

            accuracy_scores_ori_model_4.append(float("{:.3f}".format(accuracy_score(y_true_4,y_pred_4_ori)))) # CNN_40
            accuracy_scores_adv_model_4.append(float("{:.3f}".format(accuracy_score(y_true_4,y_pred_4))))
            
            progress.update(1)
            




    data_to_plot = [accuracy_scores_ori_model_1, accuracy_scores_adv_model_1, 
                    accuracy_scores_ori_model_2, accuracy_scores_adv_model_2, 
                    accuracy_scores_ori_model_4, accuracy_scores_adv_model_4, 
                    accuracy_scores_ori_model_3, accuracy_scores_adv_model_3]

    # data = accuracy_scores_ori_model_1
    # import numpy as np
    # data = accuracy_scores_ori_model_1
    # print(data)
    # fig = plt.figure(figsize =(10, 7))
    # # Creating plot
    # plt.boxplot(data)
    # # show plot
    # plt.show()

    plt.figure()
    bp = plt.boxplot(data_to_plot, showfliers=False, patch_artist=True)

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['CNN_20 Ori', 'CNN_20 Adv', 
                              'MLP_50 Ori', 'MLP_50 Adv',
                              'CNN_40 Ori', 'CNN_40 Adv',
                              'MLP_100 Ori', 'MLP_100 Adv'])
    
    # Label 25th and 75th percentiles
    for i in range(len(data_to_plot)):
        box = bp['boxes'][i]
        # Get the data points
        data = data_to_plot[i]
        # mean = np.mean(data)

        # Calculate percentiles
        q1, q3 = np.percentile(data, [25, 75])
        # Annotate the 25th percentile
        plt.text(i+1, q1, f'{q1:.2f}', ha='center', va='top', fontsize=8, color='blue')
        # Annotate the 75th percentile
        plt.text(i+1, q3, f'{q3:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
        # plt.text(i+1, mean, f'{mean:.2f}', ha='center', va='center', fontsize=8, color='red')

    plt.ylabel('Accuracy')
    plt.title('Accuracy Scores for Models on Original and Adversarial Images')
    plt.show()

