from tqdm import tqdm
import torch.nn.functional as F 
import torch
from torch.cuda.amp import autocast
import numpy as np

# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False, writer=None):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    target_bank=[]
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            meta = None
            if type(target) is list: #Will not crash if additional target information is provided.
                meta = target[1:]
                target=target[0] #Class will always be the first element of the tuple.
                
            with autocast():
                feature = net(data.cuda(non_blocking=True))
                feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(target_bank, dim=0).t().contiguous().cuda()
        # [N]
        #feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)

        test_embeddings = []
        test_targets = []
        test_sequence_ids = []
        meta = None
        #all_metas = []
        for data, target in test_bar:
            if type(target) is list: #Will not crash if additional target information is provided.
                meta = target[1:]
                target=target[0] #Class will always be the first element of the tuple.
                
            with autocast():
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
            
                #for 
                
            test_embeddings.append(feature)
            test_targets+=list(target.cpu().numpy())

            if meta is not None:
                test_sequence_ids+=meta[0]
        if writer:
            test_embeddings = torch.vstack(test_embeddings)
            test_target_labels = []
            for target in test_targets:
                test_target_labels.append(test_data_loader.dataset.classes[target])
            writer.add_embedding(test_embeddings, metadata=test_target_labels, tag="test_categorical", global_step=epoch)

            if len(test_sequence_ids)>0:
                writer.add_embedding(test_embeddings, metadata=test_sequence_ids, tag="test_sequence", global_step=epoch)

            #Small copy for Jupyter Notebooks
            np.save(f"embeddings_epoch_{epoch}.npy",test_embeddings.cpu().numpy())

            info = np.vstack([np.array(test_targets),np.array(test_sequence_ids)])
            np.save(f"info_epoch_.npy", info)


    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
