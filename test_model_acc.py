import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList_idx
from tqdm import tqdm

torch.manual_seed(0)

def test_model(model_list, dataloader,dsets, dataset_name=None):
    print('Started testing on ', len(dsets), ' images')
    netF = model_list[0]
    netB = model_list[1]
    netC = model_list[2]

    correct = 0
    total = 0
    
    with torch.no_grad():
        print('Started Testing')
        iter_test = iter(dataloader)
        for _ in tqdm(range(len(dataloader))):
            data = iter_test.next()
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')

            outputs = netC(netB(netF(inputs)))
            
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print('Accuracy of the network on the {} images: {}'.format(dataset_name, accuracy))
    return accuracy, correct, total

def data_load(batch_size=64,txt_path='domain_net_data'):
    ## prepare data

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize
        ])

    dsets = {}
    dset_loaders = {}
    train_bs = batch_size
    
    if args.dset == 'office-home':
        txt_files = {'art' : f'{txt_path}/Art.txt', 
                'clipart': f'{txt_path}/Clipart.txt', 
                'product': f'{txt_path}/Product.txt',
                'realworld': f'{txt_path}/RealWorld.txt'}

    if args.dset == 'domain-net':
        txt_files = {'clipart': f'{txt_path}/clipart_test.txt',
                    'infograph': f'{txt_path}/infograph_test.txt', 
                    'painting':  f'{txt_path}/painting_test.txt', 
                    'quickdraw': f'{txt_path}/quickdraw_test.txt', 
                    'sketch':    f'{txt_path}/sketch_test.txt', 
                    'real':      f'{txt_path}/real_test.txt'}

    for domain, paths in txt_files.items(): 
        txt_tar = open(paths).readlines()

        dsets[domain] = ImageList_idx(txt_tar, transform=image_train())
        dset_loaders[domain] = DataLoader(dsets[domain], batch_size=train_bs, shuffle=True,drop_last=False)

    return dset_loaders, dsets

def multi_domain_avg_acc(student, test_on=None):

    '''
        Given a student model and a set of domain, this func returns the avg accuracy

        Input:
            student : Student model
            test_on : List of domain to be tested on eg ['RealWorld', 'Clipart', 'Art']
        
        Return:
            Average accuracy of all domains
    '''

    if test_on is not None:
        accuracies, correct,total = [], [], []

        for sample in test_on:
            print(f'Testing Acc on {sample}')
            test_acc,corr,tot = test_model(student, dom_dataloaders[sample], dsets[sample], dataset_name=sample)
            accuracies.append(test_acc)
            correct.append(corr)
            total.append(tot)

        avg_acc = sum(accuracies)/len(accuracies)
        combined_name = '_'.join(test_on)

        print(f'\n\n Average Accuracy on {combined_name}: {avg_acc} \n\n')
        return avg_acc
    
    else:
        raise ValueError


if __name__ == '__main__':

    # modelpathF = 'optimised_rn101_MTDA_wt/KT_MTDA/domain_net/painting_to_others/target_F_painting_rn101.pt'
    # modelpathB = 'optimised_rn101_MTDA_wt/KT_MTDA/domain_net/painting_to_others/target_B_painting_rn101.pt'
    # modelpathC = 'optimised_rn101_MTDA_wt/KT_MTDA/domain_net/painting_to_others/target_C_painting_rn101.pt'
    # test_dataset = 'clipart'
    test_dataset = ['clipart', 'product', 'realworld']
    

    # training parameters
    parser = argparse.ArgumentParser(description='Args parser for KD_MTDA')
    parser.add_argument('-b', '--batch_size', default=256, type=int,help='mini-batch size (default: 32)')
    parser.add_argument('-a', '--arch', default='rn50', type=str,help='Select model type vit or rn50 based')
    parser.add_argument('-t', '--txt_path', default='data/domain_net', type=str,help='Path to all txt files')
    parser.add_argument('-d', '--dset', default='domain-net', type=str,help='Dataset')
    parser.add_argument('-n', '--bottleneck_dim', default=1024, type=str,help='Select Bottleneck dim')
    
    args = parser.parse_args()
    
    # if args.arch == 'rn50':
    #     netF = network.ResBase(res_name='resnet50').cuda()
    
    # if args.arch == 'rn101':
    #     netF = network.ResBase(res_name='resnet101').cuda()
        
    # if args.arch == 'vit':

    #     netF = network.ViT().cuda()

    # netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=args.bottleneck_dim).cuda()
    # netC = network.feat_classifier(type='wn', class_num=345, bottleneck_dim=args.bottleneck_dim).cuda()

    # netF.load_state_dict(torch.load(modelpathF))
    # netB.load_state_dict(torch.load(modelpathB))
    # netC.load_state_dict(torch.load(modelpathC))
    
    modelpath = './checkpoint/ckpt.t70_0'
    model = torch.load(modelpath)

    netF = model['netF']
    netB = model['netB']
    netC = model['netC']

    print('Models Loaded Successfully')
    netF.eval()
    netB.eval()
    netC.eval()

    model_list = [netF,netB, netC]

    dom_dataloaders, dsets = data_load(batch_size=args.batch_size,txt_path=args.txt_path) 

    if type(test_dataset) == list:
        print(f'Started testing on {test_dataset}')
        avg_acc = multi_domain_avg_acc(model_list, test_on=test_dataset)
    
    if type(test_dataset) == str:
        print(f'Started testing on {test_dataset}')
        accuracy, correct, total = test_model(model_list, dom_dataloaders[test_dataset],dsets[test_dataset], dataset_name=test_dataset)
    
    ## For testing on multiple Domains
    # avg_acc = multi_domain_avg_acc(model_list, test_on=['infograph', 'clipart', 'quickdraw', 'painting', 'real'])
    
    ## For testing on single Domain
    # accuracy, correct, total = test_model(model_list, dom_dataloaders[test_dataset],args, dataset_name=test_dataset)