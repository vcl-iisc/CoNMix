from sklearn import svm
import argparse
import os
import os.path as osp
import torchvision
import numpy as np
import torch
from torchvision import transforms
from tqdm.std import tqdm
from wandb.sdk.lib import disabled
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_dom_dis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)
    return s

def load_model(args):
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, se=False, nl=False).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000
    
    # summary(netF, (3, 224, 224))
        
    netB = network.feat_bootleneck(type="bn", feature_dim=netF.in_features,
                                   bottleneck_dim=256).cuda()
   

    modelpath = f'{args.weights_path}/{args.dset}/{names[args.s][0].upper()}{names[args.t][0].upper()}/target_F_par_0.2.pt'
    netF.load_state_dict(torch.load(modelpath))
    print('Model Loaded from', modelpath)

    netF.eval()
    netB.eval()
    # netC.eval()
    return netF,netB

def data_load(args):

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        if not alexnet:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return  transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_src = open(args.t_dset_path).readlines()

    dsets["source"] = ImageList_dom_dis(txt_src, transform=image_train(), strong_aug=0)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,drop_last=False)

    dsets["target"] = ImageList_dom_dis(txt_src, transform=image_train(), strong_aug=1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True,drop_last=False)

    return dsets, dset_loaders

def feature_extractor(loader, netF,netB):
    start_test = True
    features = []
    label =  []

    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader))):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.int()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    return all_fea, all_label


def store_feats(args):
    for net_use, weight_path in  [('resnet50','rn50/STDA_wt_fbnm_with_grad_rlcc_soft_with_stg/STDA'),('deit_s', 'weights/STDA_wt_fbnm_rlccsoft/STDA')]:
        args.net = net_use 
        args.weights_path = weight_path   
        for i in range(len(names)):
            args.s = i
            for i in range(len(names)):
                args.t = i
                if i == args.s:
                    continue
                args.s_dset_path = args.txt_path + args.dset + '/' + names[args.s] + '.txt'
                args.t_dset_path = args.txt_path + args.dset + '/' + names[args.t] + '.txt'
                print(f'Processing  {names[args.t]}_{args.net}_{names[args.s][0].upper()}{names[args.t][0].upper()}')
                dsets, dset_loaders = data_load(args)
                netF,netB = load_model(args)

                all_feas_train, all_label_train = feature_extractor(dset_loaders['target'], netF,netB)
                save_dict = {'features': all_feas_train,
                                'labels': all_label_train}
                torch.save(save_dict,f'save_feats/{names[args.t][0]}_{names[args.s][0].upper()}{names[args.t][0].upper()}_{args.net}.pth')
                
                
                all_feas_train, all_label_train = feature_extractor(dset_loaders['source'], netF,netB)
                save_dict = {'features': all_feas_train,
                                'labels': all_label_train}
                torch.save(save_dict,f'save_feats/{names[args.s][0]}_{names[args.s][0].upper()}{names[args.t][0].upper()}_{args.net}.pth')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--s', type=int, default=0, help="source")
    # parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--dset', type=str, default='office', help="dset")
    parser.add_argument('--net', type=str, default='deit_s', help="vgg16, resnet50, resnet101")
    parser.add_argument('--weights_path', type=str, default='weights/STDA_wt_fbnm_rlccsoft/STDA', help="vgg16, resnet50, resnet101")
    parser.add_argument('--txt_path', type=str, default='data/', help="vgg16, resnet50, resnet101")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--batch_size', type=int, default='32', help="batch size")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
    
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    
    if not os.path.exists('save_feats'):
        os.mkdir('save_feats')
    
    # store_feats(args) 
    domains = ['A','C','P','R']
    with open('a_dist.csv', 'w') as f:
        f.write('adaptation,model,accuracy\n')
    print('adaptation,model,accuracy')
    for src in domains:
        for tar in domains:
            if src == tar:
                continue

            use_model = f'{src}{tar}'
            for use_arch in ['resnet50','deit_s']:
                # print()
            
                load_stored_pt = torch.load(f'save_feats/{use_model[0]}_{use_model}_{use_arch}.pth')
                feats = load_stored_pt['features']
                labels = load_stored_pt['labels']

                load_stored_pt = torch.load(f'save_feats/{use_model[1]}_{use_model}_{use_arch}.pth')
                feats = torch.cat((feats, load_stored_pt['features']),dim=0)
                labels = torch.cat((labels, load_stored_pt['labels']),dim=0)

                # X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=0.1)
                # print(X_train.shape, X_test.shape)
                
                clf = MLPClassifier(random_state=1, max_iter=100).fit(feats, labels)
                y_pred = clf.predict(feats)
                acc = accuracy_score(y_pred, labels)
                with open('a_dist.csv', 'a') as f:
                    f.write(f'{use_model},{use_arch},{acc*100}\n')  
                
                print(f'{use_model},{use_arch},{acc*100}')
                