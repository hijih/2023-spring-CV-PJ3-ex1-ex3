import numpy as np
import torch
import logging

class BaselineRunner(object):
    def __init__(self, model, optimizer=None, loss_fn=None, metric=None, mode='train',**kwargs):
        self.model = model
        self.model_name = kwargs.get("model_name", "bl")
        self.device = kwargs.get("device", torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))

        if mode == 'train':
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.metric = metric
            self.train_scores = []
            self.dev_scores = []
            self.train_epoch_losses = []
            self.train_step_losses = []
            self.dev_losses = []
            self.best_score = 0

    def train(self, train_loader, dev_loader=None, **kwargs):

        # self.model.train()
        path = '/home/hjh/PJ3/'
        num_epochs = kwargs.get("num_epochs", 0)
        log_steps = kwargs.get("log_steps", 100)
        eval_steps = kwargs.get("eval_steps", 0)
        name = kwargs.get("name", "")
        num_training_steps = num_epochs * len(train_loader)

        if eval_steps:
            if self.metric is None:
                raise RuntimeError('Error: Metric can not be None!')
            if dev_loader is None:
                raise RuntimeError('Error: dev_loader can not be None!')

        global_step = 0

        for epoch in range(num_epochs):
            total_loss, total_score = 0, 0
            for X, y in train_loader:

                X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                
                if self.model_name == 'pro':
                    # print(X.device)
                    _, logits = self.model(X)
                else:
                    logits = self.model(X)
                # print(logits.device)
                # print(y.device)
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = logits.max(1)
                score = predicted.eq(y).sum().item() / len(y)
                total_score += score
                
                # self.train_step_losses.append((global_step,loss.item()))

                if log_steps and global_step%log_steps==0:
                    print(f"[Train] epoch: {epoch}/{num_epochs}, acc: {score:.5f}, loss: {loss.item():.5f}")
                    logging.info("[Train] epoch: %d / %d, acc: %.5f, loss: %.3f",epoch, num_epochs, score, loss.item())

                if eval_steps>0 and global_step>0 and \
                    (global_step%eval_steps == 0 or global_step==(num_training_steps-1)):

                    dev_score, dev_loss = self.evaluate(dev_loader, global_step=global_step)
                    # print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}") 
                    
                    if dev_score > self.best_score:
                        self.save_model(path+name+'/model.pdparams')
                        print(f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        logging.info("[Evaluate] best accuracy performence has been updated: %.5f -> %.5f", self.best_score, dev_score)
                        self.best_score = dev_score

                global_step += 1
                torch.cuda.empty_cache()
            
            trn_loss = (total_loss / len(train_loader))
            trn_score = (total_score / len(train_loader))
            self.train_epoch_losses.append(trn_loss)
            self.train_scores.append(trn_score)

        np.save(path+name+'/train_loss.npy', np.asarray(self.train_epoch_losses))
        np.save( path+name+'/dev_loss.npy', np.asarray(self.dev_losses))
        np.save(path+name+'/train_score.npy', np.asarray(self.train_scores))
        np.save(path+name+'/dev_score.npy', np.asarray(self.dev_scores))

        print("[Train] Training done!")
        logging.info("[Train] Training done!")

    with torch.no_grad():
        def evaluate(self, dev_loader, **kwargs):
            assert self.metric is not None
            self.model.eval()
            total_loss = 0
            total_score, num_count = 0, 0

            for X, y in dev_loader:

                X = X.to(self.device)
                y = y.to(self.device)

                if self.model_name == 'pro':
                    _, logitss = self.model(X)
                else:
                    logitss = self.model(X)

                loss = self.loss_fn(logitss, y)
                total_loss += loss.item()
                _, predicted = logitss.max(1)
                score = predicted.eq(y).sum().item()
                total_score += score
                batch_count = len(y)
                num_count += batch_count

            if num_count == 0:
                dev_score = 0
            else:
                dev_score = total_score / num_count

            dev_loss = (total_loss/len(dev_loader))

            self.dev_losses.append(dev_loss)
            self.dev_scores.append(dev_score)
            
            return dev_score, dev_loss
        
    with torch.no_grad():
        def predict(self, x, label=None):

            if self.model_name == 'pro':
                    _, logits = self.model(x)
            else:
                    logits = self.model(x)
            preds = torch.argmax(logits.to(self.device), axis=1).to(torch.int16)

            # if label!=None:
            #     score =  (preds.cpu() == label.to(torch.int16)).to(self.device).sum().item() / len(label).to(self.device)
            #     loss = self.loss_fn(logits.to(torch.float32).to(self.device), label.to(torch.long).to(self.device))
            #     return preds, score, loss
            # else:
            return preds

        def save_model(self, save_path):
            torch.save(self.model.state_dict(), save_path)

        def load_model(self, model_path):
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)

class SimCLRRunner(object):
    def __init__(self, model, optimizer, loss_fn, mode='train',k=512,**kwargs):
        self.model = model
        self.device = kwargs.get("device", torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
        self.temperature = kwargs.get("temperature", 0.5)
        self.num_class = kwargs.get("num_class", 100)
        self.batch_size = 0
        if mode == 'train':
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.train_scores = []
            self.dev_score1 = []
            self.dev_score5 = []
            self.train_epoch_losses = []
            self.train_step_losses = []
            self.best_score1, self.best_score5 = 0, 0
            self.num_epochs = 0
            self.k = k

    def train(self, train_loader, dev_loader=None, origin_loader=None, **kwargs):

        # self.model.train()
        path = '/home/hjh/PJ3/'
        self.num_epochs = kwargs.get("num_epochs", 0)
        log_steps = kwargs.get("log_steps", 100)
        eval_steps = kwargs.get("eval_steps", 0)
        name = kwargs.get("name", "")
        num_training_steps = self.num_epochs * len(train_loader)
        k = kwargs.get("k", 200)

        if eval_steps:
            if dev_loader is None:
                raise RuntimeError('Error: dev_loader can not be None!')
            if origin_loader is None:
                raise RuntimeError('Error: origin_loader can not be None!')
        global_step = 0

        for epoch in range(self.num_epochs):
            total_loss = 0
            for X, y in train_loader:

                trans1, trans2 = X[0].to(self.device), X[1].to(self.device)     
                self.batch_size = len(y)

                # y = y.to(self.device)
                self.optimizer.zero_grad()
                f1, logits1 = self.model(trans1)
                f2, logits2 = self.model(trans2)
                # logits = torch.cat([logits1, logits2], dim=1)
                # sim_matrix = torch.exp(torch.mm(logits, logits.t().contiguous()) / self.temperature)
                # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * len(y), device=sim_matrix.device)).bool()
                # sim_matrix = sim_matrix.masked_select(mask).view(2 * len(y), -1)
                # trans_sim = torch.exp(torch.sum(logits1 * logits2, dim=-1) / self.temperature)
                # trans_sim = torch.cat([trans_sim, trans_sim], dim=0)
                # loss = (- torch.log(trans_sim / sim_matrix.sum(dim=-1))).mean()
                loss = self.loss_fn(logits1, logits2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if log_steps and global_step%log_steps==0:
                    print(f"[Train] epoch: {epoch}/{self.num_epochs}, loss: {loss.item():.5f}")
                    logging.info("[Train] epoch: %d / %d, loss: %.3f",epoch, self.num_epochs, loss.item())

                if eval_steps>0 and global_step>0 and \
                    (global_step%eval_steps == 0 or global_step==(num_training_steps-1)):

                    dev_acc1, dev_acc5 = self.evaluate(dev_loader, origin_loader,num_class=self.num_class)
                    # print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}") 
                    if dev_acc1 > self.best_score1:
                        self.save_model(path+name+'/model_acc1.pdparams')
                        print(f"[Evaluate] best top1 accuracy performence has been updated: {self.best_score1:.5f} --> {dev_acc1:.5f}")
                        logging.info("[Evaluate] best top1 accuracy performence has been updated: %.5f -> %.5f", self.best_score1, dev_acc1)
                        self.best_score1 = dev_acc1
                    if dev_acc5 > self.best_score5:
                        self.save_model(path+name+'/model_acc5.pdparams')
                        print(f"[Evaluate] best top5 accuracy performence has been updated: {self.best_score5:.5f} --> {dev_acc5:.5f}")
                        logging.info("[Evaluate] best top5 accuracy performence has been updated: %.5f -> %.5f", self.best_score5, dev_acc5)
                        self.best_score5 = dev_acc5
                self.save_model(path+name+'/model.pdparams')
                global_step += 1
                torch.cuda.empty_cache()
            
            trn_loss = (total_loss / len(train_loader))
            self.train_epoch_losses.append(trn_loss)

        np.save(path+name+'/train_loss.npy', np.asarray(self.train_epoch_losses))
        np.save(path+name+'/dev_score1.npy', np.asarray(self.dev_score1))
        np.save(path+name+'/dev_score5.npy', np.asarray(self.dev_score5))

        print("[Train] Training done!")
        logging.info("[Train] Training done!")

    with torch.no_grad():
        def evaluate(self,  memory_data_loader, test_data_loader, num_class=10):
            k = self.k
            self.model.eval()
            total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
            with torch.no_grad():
                # generate feature bank

                for data, target in memory_data_loader:
                    feature, out = self.model(data.cuda(non_blocking=True))
                    feature_bank.append(feature)

                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
                feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=self.device)
                # loop test data to predict the label by weighted knn search
                
                for data, target in test_data_loader:
                    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    feature, out = self.model(data)

                    total_num += data.size(0)
                    # compute cos similarity between each feature vector and feature bank ---> [B, N]
                    sim_matrix = torch.mm(feature, feature_bank)
                    # [B, K]
                    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
                    # [B, K]
                    sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                    sim_weight = (sim_weight / self.temperature).exp()

                    # counts for each class
                    one_hot_label = torch.zeros(data.size(0) * k, num_class, device=self.device)
                    # [B*K, C]
                    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).to(torch.int64), value=1.0)
                    # weighted score ---> [B, C]
                    pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, num_class) * sim_weight.unsqueeze(dim=-1), dim=1)
                    pred_labels = pred_scores.argsort(dim=-1, descending=True)
                    total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
                    total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
       
                    dev_acc1 = total_top1 / total_num
                    self.dev_score1.append(dev_acc1)
                    dev_acc5 = total_top5 / total_num
                    self.dev_score5.append(dev_acc5)

            return dev_acc1, dev_acc5
        
    with torch.no_grad():
        def predict(self, x, label=None):
            logits = self.model.forward(x)
            preds = torch.argmax(logits, axis=1).to(torch.int16)

            if label!=None:
                score =  (preds == label.to(torch.int16)).sum().item() / len(label)
                loss = self.loss_fn(logits.to(torch.float32), label.to(torch.long))
                return preds, score, loss
            else:
                return preds.numpy()[0]

        def save_model(self, save_path):
            torch.save(self.model.state_dict(), save_path)

        def load_model(self, model_path):
            model_state_dict = torch.load(model_path)
            self.model.load_state_dict(model_state_dict)

            