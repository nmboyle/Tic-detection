import comet_ml
comet_ml.init(project_name='DET0102_TNet_CrossVal')

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import datetime
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from create_video_table import get_data_table_DET0102
from ssd1d import *
from data_preparation import * 
from cal_map import *
from anchors import *
from loss import DetLoss
from seg_loss import BinaryFocalLossWithLogits, BatchSoftDiceWithLogitsLoss
from seg_cal_acc import GLOBAL_cal_seg_metrics
from config import config
from tensorboardX import SummaryWriter

from lr_scheduler_and_warmup import get_lr_at_epoch, set_lr
from functools import partial

torch.backends.cudnn.benchmark = True

args = sys.argv 

test_name = args[1]
all_video_names = ['v1_hi', 'v1_lo', 'v1_no',
                   'v2_hi', 'v2_lo', 'v2_no',
                   'v3_hi', 'v3_lo', 'v3_no',
                   'v4_hi', 'v4_lo', 'v4_no']
train_video_names = [n for n in all_video_names if test_name not in n]
test_video_names = [n for n in all_video_names if test_name in n]
print(train_video_names)
print(test_video_names)

csae = [test_name not in n for n in all_video_names]

usecols = []
usecols.extend(['frame',' timestamp',' confidence',' success',' gaze_angle_x',' gaze_angle_y',
                ' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz'])
AU_i = [' AU01_r',' AU02_r',' AU04_r',' AU05_r',' AU06_r',' AU07_r',' AU09_r',' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',' AU20_r',' AU23_r',' AU25_r',' AU26_r',' AU45_r']
AU_p = [' AU01_c',' AU02_c',' AU04_c',' AU05_c',' AU06_c',' AU07_c',' AU09_c',' AU10_c',' AU12_c',' AU14_c',' AU15_c',' AU17_c',' AU20_c',' AU23_c',' AU25_c',' AU26_c',' AU28_c',' AU45_c']
usecols.extend(AU_i)
usecols.extend(AU_p)

v1_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_HI.csv',
                                label_path='./Annots/DET0102_V1_HI.csv', 
                                usecols=usecols, cut_start_and_end=csae[0])
v1_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_LO.csv',
                                label_path='./Annots/DET0102_V1_LO.csv', 
                                usecols=usecols, cut_start_and_end=csae[1])
v1_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V1_NO.csv',
                                label_path='./Annots/DET0102_V1_NO.csv', 
                                usecols=usecols, cut_start_and_end=csae[2])
v2_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_HI.csv',
                                label_path='./Annots/DET0102_V2_HI.csv', 
                                usecols=usecols, cut_start_and_end=csae[3])
v2_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_LO.csv',
                                label_path='./Annots/DET0102_V2_LO.csv', 
                                usecols=usecols, cut_start_and_end=csae[4])
v2_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V2_NO.csv',
                                label_path='./Annots/DET0102_V2_NO.csv', 
                                usecols=usecols, cut_start_and_end=csae[5])
v3_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_HI.csv',
                                label_path='./Annots/DET0102_V3_HI.csv', 
                                usecols=usecols, cut_start_and_end=csae[6])
v3_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_LO.csv',
                                label_path='./Annots/DET0102_V3_LO.csv', 
                                usecols=usecols, cut_start_and_end=csae[7])
v3_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V3_NO.csv',
                                label_path='./Annots/DET0102_V3_NO.csv', 
                                usecols=usecols, cut_start_and_end=csae[8])
v4_hi = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_HI.csv',
                                label_path='./Annots/DET0102_V4_HI.csv', 
                                usecols=usecols, cut_start_and_end=csae[9])
v4_lo = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_LO.csv',
                                label_path='./Annots/DET0102_V4_LO.csv', 
                                usecols=usecols, cut_start_and_end=csae[10])
v4_no = get_data_table_DET0102(au_table_path='./AU/DET0102_V4_NO.csv',
                                label_path='./Annots/DET0102_V4_NO.csv', 
                                usecols=usecols, cut_start_and_end=csae[11])

CAL_MAP_TABLE = {n:eval(n) for n in test_video_names}

CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
FEATURES = AU_i
# sliding window generate samples
all_train_X = []
all_train_general_labels = []
all_train_gt_boxes = []
all_train_neg_boxes = []
all_train_gt_boxes_names = []
all_train_global_gt_boxes = {}
for data_name in train_video_names:
    print(data_name)
    data = eval(data_name)
    train_all_X = data[FEATURES].values.astype(np.float32)
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values
    general_labels = data['general_labels'].values
    
    train_n_samples, train_X, train_general_labels, train_gt_boxes, train_neg_boxes = get_feat_and_gt_boxes_with_general_labels(
                                                                                            train_all_X, label_matrix, general_labels,
                                                                                            training_stride=50, sample_length=416)

    n_samples, train_gt_boxes_names, train_global_gt_boxes = get_global_gt_boxes_and_names(
                                                                data_name,label_matrix,frames,timestamp,'time',
                                                                training_stride=50, sample_length=416)
    
    all_train_X.append(train_X.astype(np.float32))
    all_train_general_labels.append(train_general_labels)
    all_train_gt_boxes.extend(train_gt_boxes)
    all_train_neg_boxes.extend(train_neg_boxes)
    all_train_gt_boxes_names.extend(train_gt_boxes_names)
    all_train_global_gt_boxes[data_name] = torch.from_numpy(train_global_gt_boxes.astype(np.float32))
    
    
all_train_X = np.concatenate(all_train_X)
all_train_general_labels = np.concatenate(all_train_general_labels)

all_train_gt_boxes, all_train_gt_boxes_names = modify_gt_boxes(all_train_gt_boxes, all_train_gt_boxes_names, thresh=3)
all_train_gt_boxes = [torch.from_numpy(e).float() for e in all_train_gt_boxes]
all_train_neg_boxes = modify_gt_boxes(all_train_neg_boxes, thresh=3)
all_train_neg_boxes = [torch.from_numpy(e).float() for e in all_train_neg_boxes]

all_train_X = torch.from_numpy(all_train_X).float()
all_train_general_labels = torch.from_numpy(all_train_general_labels).unsqueeze(1)


SEG_TEST_PREDEFINE = {}
all_test_X = []
all_test_general_labels = []
all_test_video_general_labels = {}
all_test_gt_boxes = []
all_test_gt_boxes_names = []
all_test_global_gt_boxes = {}
for data_name in test_video_names:
    data = eval(data_name)
    test_all_X = data[FEATURES].values.astype(np.float32)
    label_matrix = data[CLASSES].values
    frames = data['frame'].values
    timestamp = data[' timestamp'].values
    general_labels = data['general_labels'].values
    all_test_video_general_labels[data_name] = general_labels

    test_n_samples, test_X, test_general_labels, test_gt_boxes, _ = get_feat_and_gt_boxes_with_general_labels(
                                                                        test_all_X, label_matrix, general_labels,
                                                                        training_stride=50, sample_length=416)

    n_samples, test_gt_boxes_names, test_global_gt_boxes = get_global_gt_boxes_and_names(
                                                                data_name,label_matrix,frames,timestamp,'time',
                                                                training_stride=50, sample_length=416)
    
    all_test_X.append(test_X.astype(np.float32))
    all_test_general_labels.append(test_general_labels)
    all_test_gt_boxes.extend(test_gt_boxes)
    all_test_gt_boxes_names.extend(test_gt_boxes_names)
    all_test_global_gt_boxes[data_name] = torch.from_numpy(test_global_gt_boxes)
    SEG_TEST_PREDEFINE[data_name] = [len(test_gt_boxes_names), data.shape[0]]
    
all_test_X = np.concatenate(all_test_X)
all_test_general_labels = np.concatenate(all_test_general_labels)

all_test_gt_boxes, all_test_gt_boxes_names = modify_gt_boxes(all_test_gt_boxes, all_test_gt_boxes_names, thresh=3)
all_test_gt_boxes = [torch.from_numpy(e).float() for e in all_test_gt_boxes]

all_test_X = torch.from_numpy(all_test_X).float()
all_test_general_labels = torch.from_numpy(all_test_general_labels).unsqueeze(1)

# max value normalize 
all_train_X = all_train_X / 5
all_test_X = all_test_X / 5


print('==================')
print('read data done!')
print('==================')
print('Train_X:',all_train_X.shape, 'Test_X:',all_test_X.shape)

which_cuda = args[7]
os.environ['CUDA_VISIBLE_DEVICES'] = which_cuda
use_data_parallel = config['use_data_parallel']
device = torch.device('cuda:{}'.format(which_cuda[0])) if torch.cuda.is_available() else torch.device('cpu')
print(device)


BATCH_SIZE = int(args[6])

is_noise = [False, True][1]

train_dataset = TicDataset_all_aug_with_general_labels(all_train_X.permute(0,2,1), all_train_general_labels, all_train_gt_boxes, all_train_neg_boxes,
                                                        mode='training', 
                                                        is_noise=is_noise)
test_dataset = TicDataset_all_aug_with_general_labels(all_test_X.permute(0,2,1), all_test_general_labels, all_test_gt_boxes, mode='testing', 
                                                        is_noise=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_function_with_general_labels, drop_last=False,
                            pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_function_with_general_labels, drop_last=False, 
                            pin_memory=True)

print('Train batch_size = {}'.format(BATCH_SIZE))

is_focal = [False,True][0]

share = args[2]
print('det seg share encoder?', share)
if share == 'share':
    det_seg_share_encoder = True
else:
    det_seg_share_encoder = False
print(det_seg_share_encoder)

unet_nlayers = [3,5][0]
seg_ce_loss_type = ['focalBCE', ''][0]
multiple_scale_proba_vector = [False,True][0]

net = Det_Seg1D(unet_nlayers=unet_nlayers, det_seg_share_encoder=det_seg_share_encoder, seg_ce_loss_type=seg_ce_loss_type, multiple_scale_proba_vector=multiple_scale_proba_vector,
                backbone_name='resnet18_large512', device=device,
                thresh_low= -0.4, thresh_high=0.3, loss_type='DetLoss',
                topk_match=8, iou_smooth=False, is_focal=is_focal, uni_mat=False, 
                use_EIoU_loss=True, use_EIoU=True,
                conf_threshold=0.2, nms_threshold=0.2, top_k=100)

save_name = args[3]
if save_name == 'no_pretrain':
    print('No Pretrained model!')
    start_epoch = 0
    continue_last_time = False
else:
    continue_last_time = True
    pretrained_dict = torch.load('saved_models/{}'.format(save_name), map_location='cpu')
    start_epoch = int(save_name.split('_')[-1].strip('.pth')) + 1

    model_dict = net.state_dict()
    net.load_state_dict(pretrained_dict['model_state'])
    print('load model successful!')
    print(save_name)

    EXPERIMENT_KEY = os.environ.get("COMET_EXPERIMENT_KEY", None)
    experiment = comet_ml.ExistingExperiment(
                                previous_experiment=EXPERIMENT_KEY,
                                log_env_details=True, # to continue env logging
                                log_env_gpu=True,     # to continue GPU logging
                                log_env_cpu=True,     # to continue CPU logging
                                )
    # Retrieved from above APIExperiment
    step = 0
    experiment.set_step(step)
    experiment.set_epoch(start_epoch)


# device_ids = [4,5,6,7]
device_ids = [int(e) for e in which_cuda.split(',')]
# device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
print('visible devices:', os.environ['CUDA_VISIBLE_DEVICES'])
print('device id:', device_ids)

if use_data_parallel:
    net = torch.nn.DataParallel(net, device_ids=device_ids).to(device)
else:
    net.to(device)
print('use DataParallel?', use_data_parallel)
print('model moves to GPU done!')


# all_proposal_boxes = kmeans_anchors(all_train_gt_boxes)
all_proposal_boxes = customized_anchors()
all_proposal_boxes = all_proposal_boxes.to(device)
# using data_parallel will split the input, so need to cat len(device_ids) anchors
input_2_net_all_proposal_boxes = torch.cat([all_proposal_boxes]*len(device_ids))

num_epoch = 64

lr = float(args[4]) # 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.843, 0.999), weight_decay=1e-3) # scrach
if continue_last_time:
    optimizer.load_state_dict(pretrained_dict['optimizer_state'])
    print('optimizer state load!')


get_lr = partial(get_lr_at_epoch, 
                 warmup_epoch=5, highlr_epoch=15, 
                 base_lr=1e-4, high_lr=5e-4, 
                 eta_min=4e-8, T_max=10, # cos annealing
                 warmup_start_lr=1e-5) 


import logging
# log_path = 'log/debug.log'
log_path = args[5]


current_model_name = log_path.split('/')[-1].split('.log')[0]
if not os.path.exists('saved_models/{}'.format(current_model_name)):
    os.makedirs('saved_models/{}'.format(current_model_name))
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = logging.FileHandler(log_path) # log_path
log_file.setLevel(logging.INFO)
logger.addHandler(log_file)
logger.info('save path: saved_models/{}'.format(current_model_name))

not_use_comet = False
writer = SummaryWriter(comment=log_path.split('/')[-1].strip('.log'), comet_config={"disabled": not_use_comet})


min_loss = 200
last_epoch_loss = 1e-5
early_stop_count = 0
alpha_clf = 1. # 1 
alpha_reg = 2. # 2 
alpha_seg_ce = 3. # 3.
alpha_seg_dice = 3. # 3. 
IOU_THRESH = 0.5

scaler = torch.cuda.amp.GradScaler()
if continue_last_time:
    scaler.load_state_dict(pretrained_dict['scaler_state'])
    print('scaler state load!')

det_loss_func = DetLoss()
ce_loss_func = BinaryFocalLossWithLogits(alpha=0.25,gamma=2.,reduction='mean')
dice_loss_func = BatchSoftDiceWithLogitsLoss() # batchsoftdice loss, already avged

num_batch = len(train_dataset)//BATCH_SIZE
print('Training data has {} samples, {}//{}={} batch'.format(len(train_dataset),len(train_dataset),BATCH_SIZE,num_batch))
for epoch in range(start_epoch, num_epoch):
    # cur_lr = optimizer.param_groups[0]['lr']
    # print('Epoch=', epoch, ' lr=', cur_lr)

    net.train()
    t_start = time.time()
    
    cur_loss_l,cur_loss_c = 0.0,0.0
    cur_eiou_loss,cur_reg_loss = 0.0,0.0
    cur_ce_loss,cur_dice_loss = 0.0,0.0

    for i,(cur_X,cur_general_labels,cur_gt_boxes,cur_neg_boxes) in enumerate(train_loader):
        cur_lr = get_lr(epoch + float(i) / len(train_loader))
        set_lr(optimizer, cur_lr)

        print(f'epoch {epoch}, iter {i} / {len(train_loader)}, current learning rate:', optimizer.param_groups[0]['lr'])

        cur_X = cur_X.to(device)
        cur_general_labels = cur_general_labels.to(device)
        cur_gt_boxes = [e.to(device) for e in cur_gt_boxes]
        cur_neg_boxes = [e.to(device) for e in cur_neg_boxes]
        
        optimizer.zero_grad()     
        with torch.cuda.amp.autocast():              
            new_clf_preds_all, reg_preds_all, seg_results_all = net(cur_X, input_2_net_all_proposal_boxes, None, None, None)

            clf_loss, reg_loss = det_loss_func(new_clf_preds_all, reg_preds_all, all_proposal_boxes, cur_gt_boxes, neg_boxes=cur_neg_boxes, ignore_boxes=None,
                                                    device=device, neg_pos_ratio=2,
                                                    thresh_low=-0.4, thresh_high=0.3, 
                                                    is_focal=False, uni_mat=False, use_EIoU_loss=True, use_EIoU=True, use_MAE_proba_loss=False)
        
            celoss,diceloss = 0,0

            for _,cur_d in enumerate(seg_results_all):
                if seg_ce_loss_type == 'focalBCE':
                    celoss += ce_loss_func(cur_d,cur_general_labels)
                else:
                    raise NotImplementedError
                    num_pos = seg_labels.sum()
                    pos_weight = (len(seg_labels) - num_pos) / num_pos
                    # pos_weight[pos_weight == np.inf] = 0
                    self.ce_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    celoss += self.ce_loss_func(cur_d,seg_labels)
                
                diceloss += dice_loss_func(cur_d,cur_general_labels)


            rl = alpha_reg*reg_loss.mean()
            clfl = alpha_clf*clf_loss.mean()
            cel = alpha_seg_ce*celoss
            dicel = alpha_seg_dice*diceloss

            loss = rl + clfl + cel + dicel 
        # loss.backward() ###
        scaler.scale(loss).backward() ###
        
        ##
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        ##
        
        # optimizer.step() ###
        scaler.step(optimizer)
        scaler.update()

        cur_loss_l += rl.item()
        cur_loss_c += clfl.item()
        
        cur_ce_loss += cel.item() 
        cur_dice_loss += dicel.item() 

        logger.info('iter {}:({:d}%) | reg_loss:{:.4f} | clf_loss:{:.4f} | seg_ce_loss:{:.4f} | seg_dice_loss:{:.4f} | total_loss:{:.4f}'.format(
                            i, int(i/num_batch*100), rl.item(), clfl.item(), cel.item(), dicel.item(), loss.item()))

        # del clf_loss,reg_loss,celoss,diceloss
        # del rl,clfl,cel,dicel,loss

    cur_loss_l /= (i+1)
    cur_loss_c /= (i+1)

    cur_ce_loss /= (i+1)
    cur_dice_loss /= (i+1)

    cur_loss = cur_loss_l+cur_loss_c+cur_ce_loss+cur_dice_loss

    logger.info('epoch:{} | seg_ce_loss:{:.4f} | seg_dice_loss:{:.4f}'.format(epoch, cur_ce_loss, cur_dice_loss))
    writer.add_scalars('Train/Loss', {'reg loss': cur_loss_l,
                                      'clf loss': cur_loss_c,
                                      'total Seg ce loss': cur_ce_loss,
                                      'Seg ce loss 1': alpha_seg_ce*celoss.item(),
                                      'total Seg dice loss': cur_dice_loss,
                                      'Seg dice loss 1': alpha_seg_dice*diceloss.item(),
                                      'total loss': cur_loss}, epoch)
    
    if (epoch+1) % 50 == 0:
        print(log_path)
        save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
        if not os.path.exists('saved_models/{}/'.format(current_model_name)):
            os.makedirs('saved_models/{}/'.format(current_model_name))

        checkpoint = {
                "epoch": epoch,
                "model_state": net.module.state_dict() if use_data_parallel else net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                "loss": cur_loss,
            }
        
        torch.save(checkpoint,'saved_models/{}/{}.pth'.format(current_model_name,save_name))

        # eval test
        net.eval()
        all_detections = []
        all_seg_results = []
        with torch.no_grad():
            for i,(cur_X,_,_) in enumerate(test_loader):
                cur_X = cur_X.to(device)
                # cur_general_labels = cur_general_labels.to(device)
                # cur_gt_boxes = [e.to(device) for e in cur_gt_boxes]
                detections, d1 = net(cur_X,input_2_net_all_proposal_boxes,None,None)
                all_detections.append(detections)
                all_seg_results.append(torch.sigmoid(d1).detach().cpu())

            all_detections = torch.cat(all_detections)    
            _,_,test_ap,test_auc, TP,MatchedGT,_,_, num_tp, num_fp, num_fn, _,_,_,_, _ = cal_tp_fp_fn(
                                                                                            test_video_names,
                                                                                            all_detections,all_test_gt_boxes_names,all_test_global_gt_boxes,
                                                                                            iou_threshold = IOU_THRESH, CAL_MAP_TABLE=CAL_MAP_TABLE)
            
            ### seg eval
            all_seg_results = torch.cat(all_seg_results, 0)
            dice,acc,recall,prec,f1 = GLOBAL_cal_seg_metrics(all_seg_results, all_test_video_general_labels, all_test_gt_boxes_names, 
                                                             SEG_TEST_PREDEFINE, test_video_names)

        logger.info('epoch:{} | reg_loss:{:.4f} | clf_loss:{:.4f} | test_AP:{:.4f}'.format(epoch, cur_loss_l, cur_loss_c, test_ap))
        logger.info('epoch:{} | TP:{} | FP:{} | FN:{}'.format(epoch, num_tp, num_fp, num_fn))
        logger.info('epoch:{} | dice:{:.4f} | acc:{:.4f} | recall:{:.4f} | precision:{:.4f} | f1:{:.4f}'.format(epoch, dice,acc,recall,prec,f1))
        
        writer.add_scalars('Test/metrics', {'AP': test_ap, 
                                            'TP#':num_tp, 'FP#':num_fp, 'FN#':num_fn,
                                            'dice':dice, 'acc':acc, 'recall':recall, 'precision':prec, 'f1':f1}, epoch)
        
    else:
        logger.info('epoch:{} | reg_loss:{:.4f} | clf_loss:{:.4f}'.format(epoch, cur_loss_l, cur_loss_c))
    

    if cur_loss < min_loss:
        print('new min loss!')
        save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
        if not os.path.exists('saved_models/{}/'.format(current_model_name)):
            os.makedirs('saved_models/{}/'.format(current_model_name))

        checkpoint = {
                "epoch": epoch,
                "model_state": net.module.state_dict() if use_data_parallel else net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                "loss": cur_loss,
            }
        
        torch.save(checkpoint,'saved_models/{}/{}.pth'.format(current_model_name,save_name))

        # print('model_saved')
        min_loss = cur_loss
        ##------------------------------------------------------------------ eval test
        net.eval()
        all_detections = []
        all_seg_results = []
        with torch.no_grad():
            for i,(cur_X,_,_) in enumerate(test_loader):
                cur_X = cur_X.to(device)
                detections, d1 = net(cur_X,input_2_net_all_proposal_boxes,None,None)
                all_detections.append(detections)
                all_seg_results.append(torch.sigmoid(d1).detach().cpu())

            all_detections = torch.cat(all_detections)
            print('start eval!')
            try:
                _,_, test_ap,test_auc, TP,MatchedGT,_,_, num_tp, num_fp, num_fn,_,_,_,_,_ = cal_tp_fp_fn(test_video_names,
                                                                                        all_detections,all_test_gt_boxes_names,all_test_global_gt_boxes,
                                                                                        iou_threshold = IOU_THRESH)
            
                print('end eval!')

                ### seg eval
                all_seg_results = torch.cat(all_seg_results, 0)
                dice,acc,recall,prec,f1 = GLOBAL_cal_seg_metrics(all_seg_results, all_test_video_general_labels, all_test_gt_boxes_names, 
                                                                 SEG_TEST_PREDEFINE, test_video_names)

                ##------------------------------------------------------------------ eval test
                writer.add_scalars('Test/metrics', {'AP': test_ap, 
                                                    'TP#':num_tp, 'FP#':num_fp, 'FN#':num_fn,
                                                    'dice':dice, 'acc':acc, 'recall':recall, 'precision':prec, 'f1':f1}, epoch)
                
                logger.info('epoch:{} | dice:{:.4f} | acc:{:.4f} | recall:{:.4f} | precision:{:.4f} | f1:{:.4f}'.format(epoch, dice,acc,recall,prec,f1))

                logger.info('model saved! saved_models/{}/{}.pth. AP:{:.4f} or {:4f}'.format(current_model_name,save_name,test_ap,test_auc))
            
            except RuntimeError as e:
                print('eval failed!')
                print(e)
                logger.info('model saved! saved_models/{}/{}.pth. Eval failed, no metrics!!!!!'.format(current_model_name,save_name))

        
    if cur_loss > last_epoch_loss:
        early_stop_count += 1
    else:
        early_stop_count = 0
    if early_stop_count >= 10:
        print('early stop!')
        writer.close()
        break
    
    last_epoch_loss = cur_loss
    
    if epoch == num_epoch-1:
        print('Last epoch cost {:.2f} s'.format(time.time()-t_start))
    

save_name = '{}_{}_epoch_{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),current_model_name,epoch)
checkpoint = {
        "epoch": epoch,
        "model_state": net.module.state_dict() if use_data_parallel else net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        "loss": cur_loss,
    }

torch.save(checkpoint,'saved_models/{}/{}.pth'.format(current_model_name,save_name))

print('model_saved: Last model')

writer.close()

### args = sys.argv 
# args[0]: this file's name
# args[1]: test session, v1
# args[2]: share encoder, not_share
# args[3]: pretrained model, 'xx/xx/xx.pth'
# args[4]: learning rate, 1e-4
# args[5]: log path
# args[6]: batch size
# args[7]: gpu no, 0

# python train_det0102_det_seg.py v2 share no_pretrain 1e-4 log/LOSO_v2.log 384 0

