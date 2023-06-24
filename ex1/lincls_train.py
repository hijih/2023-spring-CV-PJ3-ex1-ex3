import torch
import logging
import os
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from simclr import SimCLR
from runner import BaselineRunner
from sklearn.metrics import accuracy_score, classification_report

path = '/home/hjh/PJ3/'
mode = 'train'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_epoch = 100
batch_size = 256
lr = 0.0001
log_steps = 100
eval_steps = 100
model_name = '/trainset_pretrain_lr=0.0001_bs=128_num=40_pro_size=128'
model_path = '/home/hjh/PJ3'+model_name+'/model.pdparams'
num_classes=10
pro_size=128
cuda = True

name = 'nnew_lincs_lr='+str(lr)+'_bs='+str(batch_size)+'_num='+str(num_epoch)+model_name
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

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

model = SimCLR(num_classes=num_classes, pro_size=pro_size).to(device=device)

for layer_name, param in model.named_parameters():
    if layer_name not in ["projection.3.weight", "projection.3.bias"]:
        param.requires_grad = False
for layer_name, module in model.named_modules():
    if layer_name == 'projection.3':
        module.weight.data.normal_(mean=0.0, std=0.01)
        module.bias.data.zero_()

model_state_dict = torch.load(model_path)

for k in list(model_state_dict.keys()):
    if k.startswith("encoder") or k.startswith("projection.0"): 
        pass
    else:
        del model_state_dict[k]

msg = model.load_state_dict(model_state_dict, strict=False)
assert set(msg.missing_keys) == {"projection.3.weight", "projection.3.bias"}
print("loaded pretrained model")

un_model = torch.nn.DataParallel(model,device_ids=[1])
parameters = list(filter(lambda p: p.requires_grad, un_model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias
# optimizer = optim.Adam(parameters, lr=lr)

un_loss_fc = torch.nn.CrossEntropyLoss()
un_metric = accuracy_score
# un_optimizer = optim.SGD(parameters, lr=lr)
un_optimizer = optim.Adam(parameters, lr=lr)

runner = BaselineRunner(un_model, un_optimizer, un_loss_fc, un_metric, device=device, model_name="pro")

if mode == 'train':
    logging.info("begin training:")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    runner.train(train_loader, test_loader, num_epochs=num_epoch, log_steps=log_steps, 
                    eval_steps=eval_steps, name = name)
elif mode == 'predict':
    runner.load_model(model_path = '/home/hjh/PJ3/nnew_lincs_lr=0.0001_bs=256_num=100/trainset_pretrain_lr=0.0001_bs=128_num=40_pro_size=128/model.pdparams')
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=2000, shuffle=True)
        i=1
        for X_test, y_test in test_loader:
            if i==1:
                preds = runner.predict(X_test, y_test)
                print(classification_report(preds.cpu().detach().numpy(), y_test.cpu().detach().numpy(),zero_division=1))
            i+=1
            # classification_report(preds.cpu().detach().numpy(), y_test.cpu().detach().numpy(),zero_division=1)