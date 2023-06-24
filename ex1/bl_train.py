import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import sys
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from sklearn.metrics import accuracy_score, classification_report
from runner import BaselineRunner

cuda = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

mode = 'train'
path = '/home/hjh/PJ3/'

num_epoch = 100
batch_size = 128
lr = 0.0001
log_steps = 100
eval_steps = 100

name = 'CIFAR10_bl_lr='+str(lr)+'_bs='+str(batch_size)+'_num='+str(num_epoch)
if not os.path.exists(os.path.join(path, name)):
    os.makedirs(os.path.join(path, name))

logging.basicConfig(filename='/home/hjh/PJ3/'+name+'/log.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

bl_model = models.resnet18(num_classes = 10).to(device)
bl_optimizer = optim.Adam(lr=lr, params=bl_model.parameters())
bl_loss_fc = torch.nn.CrossEntropyLoss()
bl_metric = accuracy_score
# bl_scheduler = StepLR(bl_optimizer, step_size=5, gamma=0.1)
# bl_optimizer = optim.SGD(bl_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
# scheduler = MultiStepLR(bl_optimizer, milestones=[81, 122], gamma=0.1)

runner = BaselineRunner(bl_model, bl_optimizer, bl_loss_fc, bl_metric, device=device, model_name="bl")
if mode == 'train':
    logging.info("begin training:")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    runner.train(train_loader, test_loader, num_epochs=num_epoch, log_steps=log_steps, 
                    eval_steps=eval_steps, name = name)
elif mode == 'predict':
    runner.load_model(model_path = '/home/hjh/PJ3/bl_model.pdparams')
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=16)
        for X_test, y_test in test_loader:
            preds = runner.predict(X_test, y_test)
            print(classification_report(preds.cpu().detach().numpy(), y_test.cpu().detach().numpy(),zero_division=1))

    

