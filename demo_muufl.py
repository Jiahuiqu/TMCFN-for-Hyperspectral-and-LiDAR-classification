
import sys
print("\n".join(sys.path))
sys.path.insert(0, './keops')
# sys.path.insert(0, './')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1, 2,3'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
os.environ["USE_KEOPS"] = "False"

import numpy as np
import torch
from argparse import ArgumentParser
from model import Model_All
import utils
import scipy.io as sio
import time
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
import random
import torch.nn.functional as F
from sklearn.metrics import classification_report, recall_score, cohen_kappa_score, accuracy_score
import torch.nn as nn




def calculate_similarity(image_features, text_features, logit_scale):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # logit_scale = logit_scale.exp()
    # logit_scale = torch.exp(logit_scale)

    ## cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.T
    return logits_per_image


class ClipLoss(nn.Module):

    def __init__(self, logit_scale=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  #####

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        ## cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2

        return total_loss


class MyDataset(Dataset):
    def __init__(self, gt_labeled, All_labeled_Patch_HSI, All_labeled_Patch_LiDAR, train_mask, test_mask, batch_data_id, unlabeled_Patch_HSI, unlabeled_Patch_LiDAR):
        super(MyDataset, self).__init__()
        self.gt_labeled = gt_labeled
        self.All_labeled_Patch_HSI = All_labeled_Patch_HSI
        self.All_labeled_Patch_LiDAR = All_labeled_Patch_LiDAR
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.batch_data_id = batch_data_id
        self.unlabeled_Patch_HSI = unlabeled_Patch_HSI
        self.unlabeled_Patch_LiDAR = unlabeled_Patch_LiDAR

    def __getitem__(self, idx):
        return torch.from_numpy(self.gt_labeled[idx] - 1), torch.from_numpy(self.All_labeled_Patch_HSI[idx]), torch.from_numpy(self.All_labeled_Patch_LiDAR[idx]),torch.from_numpy(self.train_mask[idx]), \
               torch.from_numpy(self.train_mask[idx] + self.test_mask[idx]), torch.from_numpy(self.batch_data_id[idx]), torch.from_numpy(self.unlabeled_Patch_HSI[idx]), torch.from_numpy(self.unlabeled_Patch_LiDAR[idx])

    def __len__(self):
        return len(self.gt_labeled)


def get_data(hsi, lidar, gt, graph_num, patchsize_HSI, patchsize_LiDAR, unlabeled_samples_num=None, num_samples_per_class=None, ratio=None ):

    pos_list = []
    id_per_class = [[] for i in range(labels.max())]
    [pos_h, pos_w] = np.where(gt != 0)
    for pos in zip(pos_h, pos_w):
        pos_list.append(pos)
        id_per_class[gt[pos[0], pos[1]] - 1].append(len(pos_list) - 1)

    batch_data = [[] for i in range(graph_num)]
    for i in range(labels.max()):
        num = len(id_per_class[i]) // graph_num
        id = id_per_class[i]
        random.shuffle(id)
        for j in range(graph_num):
            if(j == graph_num-1):
                batch_data[j].extend(id[j * num :])
            else:
                batch_data[j].extend(id[j * num : (j+1) * num])

    pos_array = np.array(pos_list)
    for i in range(graph_num):
        batch_data[i] = np.array(batch_data[i])

    gt_labeled = [[] for i in range(graph_num)]
    for i in range(graph_num):
        gt_labeled[i] = gt[pos_array[batch_data[i]][:,0], pos_array[batch_data[i]][:,1]]

    train_mask = [np.zeros_like(temp) for temp in gt_labeled]
    test_mask = [np.zeros_like(temp) for temp in gt_labeled]


    for i in range(graph_num):
        print("\n")
        for j in range(labels.max()):
            pos = np.where(gt_labeled[i] == j + 1)[0]
            if (num_samples_per_class is not None):
                num_samples_per_class = num_samples_per_class
            elif (ratio is not  None):
                num_samples_per_class = int(len(pos) * ratio)

            print(f"i:{i+1}-{j+1}-{num_samples_per_class // graph_num}")
            pos_id = random.sample(range(len(pos)), num_samples_per_class // graph_num)
            train_mask[i][np.array(pos[pos_id])] = 1

        train_mask[i] = train_mask[i].astype(np.bool_)
        test_mask[i] = ~train_mask[i]

    labeled_data_hsi = [np.empty(shape=(temp.shape[0], hsi.shape[-1]), dtype=np.float32) for temp in gt_labeled]
    labeled_data_lidar = [np.empty(shape=(temp.shape[0], lidar.shape[-1]), dtype=np.float32) for temp in gt_labeled]
    for i in range(graph_num):
        labeled_data_hsi[i] = hsi[pos_array[batch_data[i]][:,0], pos_array[batch_data[i]][:,1], :]
        labeled_data_lidar[i] = lidar[pos_array[batch_data[i]][:,0], pos_array[batch_data[i]][:,1], :]

    bands =  hsi.shape[-1]
    temp = hsi[:, :, 0]
    pad_width = np.floor(patchsize_HSI / 2)
    pad_width = np.int32(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_HSI_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = hsi[:, :, i]
        pad_width = np.floor(patchsize_HSI / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        data_HSI_pad[:, :, i] = temp2

    temp = lidar[:, :, 0]
    pad_width2 = np.floor(patchsize_LiDAR / 2)
    pad_width2 = np.int32(pad_width2)
    temp2 = np.pad(temp, pad_width2, 'symmetric')
    [h_pad, w_pad] = temp2.shape
    data_LiDAR_pad = np.empty((h_pad, w_pad, bands), dtype='float32')
    for i in range(bands):
        temp = lidar[:, :, i]
        pad_width2 = np.floor(patchsize_LiDAR / 2)
        pad_width2 = np.int32(pad_width2)
        temp2 = np.pad(temp, pad_width2, 'symmetric')
        data_LiDAR_pad[:, :, i] = temp2

    All_labeled_Patch_HSI = [np.empty((len(temp), hsi.shape[-1], patchsize_HSI, patchsize_HSI), dtype='float32') for temp in gt_labeled]
    All_labeled_Patch_LiDAR = [np.empty((len(temp), lidar.shape[-1], patchsize_LiDAR, patchsize_LiDAR), dtype='float32') for temp in gt_labeled]
    for i in range(graph_num):
        ind1, ind2 = pos_array[batch_data[i]][:, 0], pos_array[batch_data[i]][:, 1]
        ind3 = ind1 + pad_width  #
        ind4 = ind2 + pad_width
        for j in range(len(ind1)):
            patch = data_HSI_pad[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                    (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]
            patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
            All_labeled_Patch_HSI[i][j,:,:,:] = patch
            # --------------------------------------------------------------
            patch = data_LiDAR_pad[(ind3[j] - pad_width2):(ind3[j] + pad_width2 + 1),
                    (ind4[j] - pad_width2):(ind4[j] + pad_width2 + 1), :]
            patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, bands))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (bands, patchsize_LiDAR, patchsize_LiDAR))
            All_labeled_Patch_LiDAR[i][j, :, :, :] = patch

    pos_x, pos_y = np.where(gt == 0)
    pos_id = random.sample(range(len(pos_x)), unlabeled_samples_num)

    unlabeled_Patch_HSI = np.empty((len(pos_id), hsi.shape[-1], patchsize_HSI, patchsize_HSI), dtype='float32')
    unlabeled_Patch_LiDAR = np.empty((len(pos_id), lidar.shape[-1], patchsize_LiDAR, patchsize_LiDAR), dtype='float32')

    ind1, ind2 = pos_x[pos_id], pos_y[pos_id]
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for j in range(len(ind1)):
        patch = data_HSI_pad[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]
        patch = np.reshape(patch, (patchsize_HSI * patchsize_HSI, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_HSI, patchsize_HSI))
        unlabeled_Patch_HSI[j, :, :, :] = patch
        # --------------------------------------------------------------
        patch = data_LiDAR_pad[(ind3[j] - pad_width2):(ind3[j] + pad_width2 + 1),
                (ind4[j] - pad_width2):(ind4[j] + pad_width2 + 1), :]
        patch = np.reshape(patch, (patchsize_LiDAR * patchsize_LiDAR, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize_LiDAR, patchsize_LiDAR))
        unlabeled_Patch_LiDAR[j, :, :, :] = patch

    return gt_labeled, labeled_data_hsi, labeled_data_lidar, All_labeled_Patch_HSI, All_labeled_Patch_LiDAR, train_mask, test_mask, pos_array, batch_data, unlabeled_Patch_HSI, unlabeled_Patch_LiDAR


def train(labels, text):
    opti = torch.optim.Adam([{'params': model.vision_net.parameters(), 'lr': 0.0005}])
    epoches = 1000
    warm_up = False
    warm_up_epoch = 0
    ## ----------------------------------------------------------------------------------------------------------------
    model.train()
    OA_best = 0
    AA_OA_best = 0
    Kappa_OA_best = 0
    avg_accuracy = [None] * len(train_loader)
    Loss = ClipLoss(logit_scale=5)

    for epoch in range(epoches):
        model.train()
        opti.zero_grad()
        for i, (label, patch_hsi_all, patch_lidar_all, train_mask_,test_mask_, _, unlabeled_Patch_HSI, unlabeled_Patch_LiDAR) in enumerate(train_loader):
            loss_clip_all = 0

            labeled_mask = torch.squeeze(train_mask_ + test_mask_)
            train_mask_ = torch.squeeze(train_mask_)
            len_labeled = torch.squeeze(patch_hsi_all).shape[0]
            patch_hsi = torch.cat([torch.squeeze(patch_hsi_all), torch.squeeze(unlabeled_Patch_HSI[0])], dim=0)
            patch_lidar = torch.cat([torch.squeeze(patch_lidar_all), torch.squeeze(unlabeled_Patch_LiDAR[0])], dim=0)

            Z_vison, Z_text, loss_vision_CL, logprobs = model( patch_hsi=patch_hsi, patch_lidar=patch_lidar,labeled_mask=labeled_mask, text=text, warm_up=warm_up, len_labeled=len_labeled)

            train_feature = Z_vison[train_mask_]
            train_pred = torch.nn.functional.softmax(Z_vison @ Z_text.t(), dim=1)[train_mask_].argmax(-1) # device2

            label = torch.squeeze(label).to(device2)
            train_lab = label[train_mask_]

            dim_feature = Z_vison.shape[-1]
            capacity = len(train_lab)  // class_num
            queue_0 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_1 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_2 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_3 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_4 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_5 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_6 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_7 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_8 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_9 = utils.Queue(capacity=capacity, dim=dim_feature)
            queue_10 = utils.Queue(capacity=capacity, dim=dim_feature)

            for n, label in enumerate(train_lab):
                if (label == 0):
                    queue_0.enqueue(train_feature[n,:])
                elif (label == 1):
                    queue_1.enqueue(train_feature[n,:])
                elif (label == 2):
                    queue_2.enqueue(train_feature[n,:])
                elif (label == 3):
                    queue_3.enqueue(train_feature[n,:])
                elif (label == 4):
                    queue_4.enqueue(train_feature[n,:])
                elif (label == 5):
                    queue_5.enqueue(train_feature[n,:])
                elif (label == 6):
                    queue_6.enqueue(train_feature[n,:])
                elif (label == 7):
                    queue_7.enqueue(train_feature[n,:])
                elif (label == 8):
                    queue_8.enqueue(train_feature[n,:])
                elif (label == 9):
                    queue_9.enqueue(train_feature[n,:])
                elif (label == 10):
                    queue_10.enqueue(train_feature[n,:])

            for n in range(capacity):
                temp = torch.empty((1, dim_feature))
                temp = torch.unsqueeze(queue_0.dequeue(),dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_1.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_2.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_3.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_4.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_5.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_6.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_7.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_8.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_9.dequeue(), dim=0)], dim=0)
                temp = torch.cat([temp, torch.unsqueeze(queue_10.dequeue(), dim=0)], dim=0).to(device2)
                loss_clip_all += Loss(image_features=temp, text_features=Z_text)

            logprobs = logprobs[:, train_mask_, :, :]
            graph_loss = 0.
            if logprobs is not None and epoch + 1 > warm_up_epoch:
                corr_pred = (torch.unsqueeze(train_pred, dim=0) == torch.unsqueeze(train_lab, dim=0)).float().detach()
                wron_pred = (1 - corr_pred)
                if avg_accuracy[i] is None:
                    avg_accuracy[i] = torch.ones_like(corr_pred) * 0.5
                point_w = (avg_accuracy[i] - corr_pred)
                if (len(logprobs.shape) == 4):
                    for m in range(logprobs.shape[-1]):
                        graph_loss += point_w * logprobs[:, :, :, m].exp().mean([-1, -2])
                graph_loss = graph_loss.mean()
                avg_accuracy[i] = avg_accuracy[i].to(corr_pred.device) * 0.95 + 0.05 * corr_pred


            loss = 0.5 * loss_vision_CL + 1.5 * loss_clip_all + graph_loss
            # loss = loss_vision_CL + loss_clip_all + graph_loss

            loss.backward()
            opti.step()
            model.vision_net.FCFLM.update_moving_average()
            corr_pred = (train_pred == train_lab).float().detach().sum()
            print(f"epoch: {epoches}-{epoch + 1}-{i} loss:{loss}  train_acc {corr_pred / len(train_lab)} ")
            fw.write(f"epoch: {epoches}-{epoch + 1}-{i} loss:{loss}  train_acc {corr_pred / len(train_lab)}\n ")

        if((epoch+1) % 5 ==0):
            model.eval()
            pred_map = torch.zeros((labels.shape[0], labels.shape[-1]))
            with torch.no_grad():
                for i, (label, patch_hsi_all, patch_lidar_all, train_mask_, test_mask_, batch_data_id_, unlabeled_Patch_HSI, unlabeled_Patch_LiDAR) in enumerate(train_loader):
                    labeled_mask = torch.squeeze(train_mask_ + test_mask_)
                    train_mask_ = torch.squeeze(train_mask_)
                    len_labeled = torch.squeeze(patch_hsi_all).shape[0]
                    patch_hsi = torch.cat([torch.squeeze(patch_hsi_all), torch.squeeze(unlabeled_Patch_HSI[0])], dim=0)
                    patch_lidar = torch.cat([torch.squeeze(patch_lidar_all), torch.squeeze(unlabeled_Patch_LiDAR[0])], dim=0)
                    Z_vison, Z_text, _ , _= model(patch_hsi=patch_hsi, patch_lidar=patch_lidar ,labeled_mask=labeled_mask, text=text, warm_up=warm_up, len_labeled=len_labeled)
                    batch_data_id_ = np.squeeze(batch_data_id_)
                    train_mask_ = torch.squeeze(train_mask_)
                    pred_temp = torch.nn.functional.softmax(Z_vison @ Z_text.t(), dim=1).argmax(-1).cpu().numpy() + 1
                    index = pos_array[batch_data_id_] ##
                    for i, (x, y) in enumerate(index):
                        pred_map[x, y] = pred_temp[i]

            ## -------------------------------------------------------------------------------------------------------------
            ## classfication report
            test_pred = pred_map[labels != 0]
            test_true = labels[labels != 0]
            OA = accuracy_score(test_true, test_pred)
            AA = recall_score(test_true, test_pred, average='macro')
            kappa = cohen_kappa_score(test_true, test_pred)
            class_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']  # muffl
            report_log = classification_report(test_true, test_pred, target_names=class_name, digits=4)
            print(report_log)
            fw.write(report_log)
            fw.write("\n")

            if (OA_best < OA):
                OA_best = OA
                AA_OA_best = AA
                Kappa_OA_best = kappa
                torch.save(model.state_dict(), f"./results/muufl/Muufl_{OA}.pth")
                sio.savemat(f'./results/muufl/muufl_{OA_best}.mat', {'data': pred_map.numpy()})
            print(f"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n\nOA_best:{OA_best}\nAA_OA_best:{AA_OA_best}\nKappa_OA_best:{Kappa_OA_best}\n")
            fw.write(f"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n\nOA_best:{OA_best}\nAA_OA_best:{AA_OA_best}\nKappa_OA_best:{Kappa_OA_best}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--data", default='Cora')
    parser.add_argument("--fold", default='0', type=int)
    parser.add_argument("--conv_layers", default=[[32, 32]], type=lambda x: eval(x))
    parser.add_argument("--dgm_layers", default=[[32 * 3, 32]], type=lambda x: eval(x))
    parser.add_argument("--fc_layers", default=[8, 8, 3], type=lambda x: eval(x))
    parser.add_argument("--pre_fc", default=[-1, 32], type=lambda x: eval(x))
    parser.add_argument("--gfun", default='sage')
    parser.add_argument("--ffun", default='sage')
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--test_eval", default=10, type=int)
    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------------------------------------------------------
    dataset_name = 'muufl'
    N_PCA = 15

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)
    patchsize_HSI = 19
    patchsize_LiDAR = 19
    graph_num = 15
    num_samples_per_class = 15 * 10

    text_non_extended = [
            'Trees.',
            'Mostly grass.',
            'Mixed ground surface.',
            'Dirt and sand.',
            'Road.',
            'Water.',
            'Building Shadow.',
            'Building.',
            'Sidewalk.',
            'Yellow curb.',
            'Cloth panels.'
            ]

    # text_extended = [
    #     '...',
    #     '...',
    #     '...',
    #     '...',
    #     '...',
    #     ...
    #     ]
    ## e.g. A sample of Mostly grass. The color of Mostly grass is green and the texture is fine and uniform. Mostly grass often grows alongside trees, roads or sidewalks, even mixed with dirt and sand.

    text = text_non_extended

    # ---------------------------------------------------------------------

    print(f"{patchsize_HSI} x {patchsize_HSI}")
    hsi, labels, lidar = utils.get_data(dataset_name=dataset_name)
    hsi, _ = utils.applyPCA(hsi, N_PCA)
    lidar = np.expand_dims(lidar, 2).repeat(N_PCA, axis=2)
    class_num = labels.max()

    gt_labeled, labeled_data_hsi, labeled_data_lidar, All_labeled_Patch_HSI, All_labeled_Patch_LiDAR, train_mask, test_mask, pos_array, batch_data_id, unlabeled_Patch_HSI, unlabeled_Patch_LiDAR = \
        get_data(hsi=hsi, lidar=lidar, gt=labels, graph_num=graph_num, patchsize_HSI=patchsize_HSI,
                 patchsize_LiDAR=patchsize_LiDAR, num_samples_per_class=num_samples_per_class,
                 unlabeled_samples_num=graph_num * 100, ratio=None)

    unlabeled_HSI_list = []
    unlabeled_LiDAR_list = []
    for i in range(graph_num):
        unlabeled_HSI_list.append(unlabeled_Patch_HSI[100 * i:100 * (i + 1)])
        unlabeled_LiDAR_list.append(unlabeled_Patch_LiDAR[100 * i:100 * (i + 1)])

    dataset = MyDataset(gt_labeled, All_labeled_Patch_HSI, All_labeled_Patch_LiDAR, train_mask, test_mask,
                        batch_data_id, unlabeled_HSI_list, unlabeled_LiDAR_list)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)  #

    model = Model_All(params, patchsize_HSI, device1=device, device2=device2, class_num=labels.max())
    model.text_net.load_state_dict(torch.load("./text_encoder.pth"))  ##

    fw = open("./log.txt", 'a+')
    fw.write(f"\n\n\n")
    fw.write(f"Time: {time.asctime( time.localtime(time.time()) )}\n")
    fw.write(f"{patchsize_HSI} x {patchsize_HSI}")
    train(labels, text)

