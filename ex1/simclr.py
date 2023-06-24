import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SimCLR(nn.Module):
    def __init__(self, num_classes=100, pro_size=64):
        super(SimCLR, self).__init__()
        self.encoder = []
        for name, module in models.resnet18(num_classes=num_classes).named_children():
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)

        self.encoder = nn.Sequential(*self.encoder)

        self.projection = nn.Sequential(
                            nn.Linear(512, pro_size),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5),
                            nn.Linear(pro_size, num_classes))
        # self.projection = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True), nn.Linear(512, num_classes, bias=True))

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projection(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class SimCLRLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(SimCLRLoss,self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self,out_1,out_2):
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

class SimCLRLoss2(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLRLoss2, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

show_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),  # with 0.5 probability
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

trans_transform = transforms.Compose([transforms.ToTensor()])

if __name__ == '__main__':
    origin_data = STL10('data', split="unlabeled", download=False, transform=trans_transform)
    origin_loader = DataLoader(origin_data, batch_size=1, shuffle=False, pin_memory=True, drop_last=True)
    i = 0
    for x,y in origin_loader:
        if i ==0:
            print(x.shape)
            plt.imshow(x[i].permute(1, 2, 0))
            plt.savefig('orgin2.jpg')
            a = show_transform(Image.fromarray(x[i].numpy()))
            b = show_transform(Image.fromarray(x[i].numpy()))
            print(a.shape,b.shape)
            a, b = a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy()
            plt.imshow(a)
            plt.savefig('trans3.jpg')
            plt.imshow(b)
            plt.savefig('trans4.jpg')
            break


