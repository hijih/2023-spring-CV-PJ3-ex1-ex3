import torch
import logging
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from simclr import SimCLR, TransformsSimCLR, SimCLRLoss2
from runner import SimCLRRunner

cuda = True
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
path = '/home/hjh/PJ3/'

num_epochs = 40
num_classes = 10
batch_size = 128
temperature = 0.5
lr = 1e-4
pro_size = 128
log_steps = 10
k = 512
eval_steps = 100

name = 'trainset_pretrain_lr='+str(lr)+'_bs='+str(batch_size)+'_num='+str(num_epochs)+'_pro_size='+str(pro_size)
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

trans = TransformsSimCLR(size=224)

train_data = STL10('/home/hjh/PJ3', split="unlabeled", download=False, transform=trans)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

pretrain_model = SimCLR(num_classes=num_classes, pro_size=pro_size).to(device=device)
pretrain_optimizer = optim.Adam(pretrain_model.parameters(), lr=lr)
pretrain_loss_fn = SimCLRLoss2(batch_size=batch_size, temperature=temperature)

runner = SimCLRRunner(pretrain_model, pretrain_optimizer, pretrain_loss_fn, temperature = temperature, 
                      num_classes=num_classes, k=k, device=device)
runner.train(train_loader, num_epochs = num_epochs, 
             log_steps = log_steps,name=name)

