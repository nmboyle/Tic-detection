from sklearn.cluster import KMeans, kmeans_plusplus
import torch
import torch.nn as nn
import numpy as np


def kmeans_anchors(all_train_gt_boxes):
    train_gt_boxes_with_obj = torch.cat([e for e in all_train_gt_boxes if e.size()[0]>0])
    length = train_gt_boxes_with_obj[:,1] - train_gt_boxes_with_obj[:,0]
    tic_center = (train_gt_boxes_with_obj[:,1] + train_gt_boxes_with_obj[:,0])/2

    # center = np.array([15.5+i*32 for i in range(8)])
    # center = np.array([15.5+i*32 for i in range(13)])

    n_clusters = 10
    original_train_gt_boxes_with_obj = train_gt_boxes_with_obj.repeat(n_clusters,1,1)

    # # kmeans++ initialize 
    w = kmeans_plusplus(length.numpy().reshape(-1, 1),n_clusters,random_state=n_clusters)[0].reshape(-1,).astype(np.float)

    w_old = w.copy()

    counter=0
    while counter==0 or any(np.abs(w_old - w) > 0.01):
        counter+=1
        w_old = w.copy()

        new_tic = np.stack((tic_center.reshape(1,-1) - w.reshape(-1,1)/2, tic_center.reshape(1,-1) + w.reshape(-1,1)/2),2)
        inter = np.clip(np.minimum(original_train_gt_boxes_with_obj[:,:,1], new_tic[:,:,1]) - np.maximum(original_train_gt_boxes_with_obj[:,:,0], new_tic[:,:,0]), a_min=0,a_max=416)
        union = length.reshape(1,-1) + w.reshape(-1,1) - inter
        dist = 1-inter/union
        assign = dist.argmin(axis=0)

        for i in range(n_clusters):
            w[i] = length[assign == i].mean()

        how_many_nan = np.isnan(w).sum()
        if how_many_nan:
            print("There isn't {} clusters to assign, only {} clusters".format(n_clusters,n_clusters-how_many_nan))

    # print(proposal_boxes.shape)
    w = np.sort(w)
    print(w)

    #----------------------------------------------------------------------------- Multiscale的anchor
    w4 = np.sort(np.concatenate((w[:3],((np.append(0,w[:3]) + np.append(w[:3],0))/2)[:-1])))
    w8 = np.sort(np.concatenate((w[1:4],((np.append(0,w[1:4]) + np.append(w[1:4],0))/2)[:-1])))
    w16 = np.sort(np.concatenate((w[2:5],((np.append(0,w[2:5]) + np.append(w[2:5],0))/2)[:-1])))
    w32 = np.sort(np.concatenate((w[4:],((np.append(0,w[4:]) + np.append(w[4:],0))/2)[:-1])))
    w_all = [w4,w8,w16,w32]
    print(w4,'\n',w8,'\n',w16,'\n',w32,'\n')
    time_length = [104,52,26,13] # 4,8,16,32
    print(time_length)
    grid_center = [1.5, 3.5, 7.5, 15.5]
    downsample = [4,8,16,32]

    all_proposal_boxes = []
    for level in range(4):
        center = np.array([grid_center[level]+i*downsample[level] for i in range(time_length[level])])
        wi = w_all[level]
        proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - wi.reshape(-1,1)/2, center.reshape(1,-1) + wi.reshape(-1,1)/2),2))
        # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
        all_proposal_boxes.append(torch.from_numpy(proposal_boxes))

    all_proposal_boxes = torch.cat(all_proposal_boxes)
    print(all_proposal_boxes.shape)
    return all_proposal_boxes

def kmeans_anchors_no_repeat(all_train_gt_boxes):
    train_gt_boxes_with_obj = torch.cat([e for e in all_train_gt_boxes if e.size()[0]>0])
    length = train_gt_boxes_with_obj[:,1] - train_gt_boxes_with_obj[:,0]
    tic_center = (train_gt_boxes_with_obj[:,1] + train_gt_boxes_with_obj[:,0])/2

    # center = np.array([15.5+i*32 for i in range(8)])
    # center = np.array([15.5+i*32 for i in range(13)])

    n_clusters = 12
    original_train_gt_boxes_with_obj = train_gt_boxes_with_obj.repeat(n_clusters,1,1)

    # # kmeans++ initialize 
    w = kmeans_plusplus(length.numpy().reshape(-1, 1),n_clusters,random_state=n_clusters)[0].reshape(-1,).astype(np.float)

    w_old = w.copy()

    counter=0
    while counter==0 or any(np.abs(w_old - w) > 0.01):
        counter+=1
        w_old = w.copy()

        new_tic = np.stack((tic_center.reshape(1,-1) - w.reshape(-1,1)/2, tic_center.reshape(1,-1) + w.reshape(-1,1)/2),2)
        inter = np.clip(np.minimum(original_train_gt_boxes_with_obj[:,:,1], new_tic[:,:,1]) - np.maximum(original_train_gt_boxes_with_obj[:,:,0], new_tic[:,:,0]), a_min=0,a_max=416)
        union = length.reshape(1,-1) + w.reshape(-1,1) - inter
        dist = 1-inter/union
        assign = dist.argmin(axis=0)

        for i in range(n_clusters):
            w[i] = length[assign == i].mean()

        how_many_nan = np.isnan(w).sum()
        if how_many_nan:
            print("There isn't {} clusters to assign, only {} clusters".format(n_clusters,n_clusters-how_many_nan))

    # print(proposal_boxes.shape)
    w = np.sort(w)
    print(w)

    #----------------------------------------------------------------------------- Multiscale的anchor
    w4 = w[:3]
    w8 = w[3:6]
    w16 = w[6:9]
    w32 = w[9:12]
    
    w_all = [w4,w8,w16,w32]
    print(w4,'\n',w8,'\n',w16,'\n',w32,'\n')
    time_length = [104,52,26,13] # 4,8,16,32
    print(time_length)
    grid_center = [1.5, 3.5, 7.5, 15.5]
    downsample = [4,8,16,32]

    all_proposal_boxes = []
    for level in range(4):
        center = np.array([grid_center[level]+i*downsample[level] for i in range(time_length[level])])
        wi = w_all[level]
        proposal_boxes = np.concatenate(np.stack((center.reshape(1,-1) - wi.reshape(-1,1)/2, center.reshape(1,-1) + wi.reshape(-1,1)/2),2))
        # proposal_boxes = torch.clamp(torch.from_numpy(proposal_boxes), 0, 415)
        all_proposal_boxes.append(torch.from_numpy(proposal_boxes))

    all_proposal_boxes = torch.cat(all_proposal_boxes)
    print(all_proposal_boxes.shape)
    return all_proposal_boxes


def customized_anchors():
    '''
    For each scale, 
    # 4,8,16
    # 16,32,64
    # 64,96,128,
    # 96,128,192,256,320,384
    '''
    ba_0 = torch.tensor([[-2.0000,  2.0000],
                         [-3.0314,  3.0314],
                        [-4.0000,  4.0000],
                        [-6.0629,  6.0629],
                        [ -8.0000, 8.0000],
                        [-12.1257,  12.1257]])
    ba_1 = torch.tensor([[ -8.0000, 8.0000],
                        [-12.1257,  12.1257],
                        [-16.0000,  16.0000],
                        [-24.2515,  24.2515],
                        [-32.0000,  32.0000],
                        [-48.5029,  48.5029]])
    ba_2 = torch.tensor([[-32.0000,  32.0000],
                        [-48.5029,  48.5029],
                        [-48.0000,  48.0000],
                        [-72.7544,  72.7544],
                        [ -64.0000,   64.0000],
                        [ -97.0059,   97.0059]])
    ba_3 = torch.tensor([[-48.0000,  48.0000],
                        [-72.7544,  72.7544],
                        [ -64.0000,   64.0000],
                        [ -97.0059,   97.0059],
                        [ -96.0000,   96.0000],
                        [-145.5088,  145.5088],
                        [-128.0000,  128.0000],
                        [-194.0117,  194.0117],
                        [-160.0000,  160.0000],
                        [-242.5146,  242.5146],
                        [-192.0000,  192.0000],
                        [-291.0176,  291.0176]])

    ba = [ba_0,ba_1,ba_2,ba_3]
    strides = [4,8,16,32]
    tsizes = [104,52,26,13]

    all_anchors = []
    for i in range(0, 4):
        shifts = torch.arange(0+0.5, tsizes[i]+0.5, device='cpu') * strides[i]
        anchors = ba[i][None, :, :] + shifts[:, None, None]
        all_anchors.append(anchors.view(-1, 2))

    return torch.cat(all_anchors)



  