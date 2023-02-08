import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from helper.data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import wandb

def op_copy(optimizer):
	for param_group in optimizer.param_groups:
		param_group['lr0'] = param_group['lr']
	return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
	decay = (1 + gamma * iter_num / max_iter) ** (-power)
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr0'] * decay
		param_group['weight_decay'] = 1e-3
		param_group['momentum'] = 0.9
		param_group['nesterov'] = True
		wandb.log({'MISC/LR': param_group['lr']})
	return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
	if not alexnet:
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	else:
		normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
	return  transforms.Compose([
		transforms.Resize((resize_size, resize_size)),
		transforms.RandomCrop(crop_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize
	])

def image_test(resize_size=256, crop_size=224, alexnet=False):
	if not alexnet:
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	else:
		normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
	return  transforms.Compose([
		transforms.Resize((resize_size, resize_size)),
		transforms.CenterCrop(crop_size),
		transforms.ToTensor(),
		normalize
	])

def data_load(args): 
	## prepare data
	dsets = {}
	dset_loaders = {}
	train_bs = args.batch_size
	# print(args.s_dset_path)
	txt_src = open(args.s_dset_path).readlines()
	txt_test = open(args.test_dset_path).readlines()

	if args.dset =='domain_net':
		txt_eval_dn = open(args.txt_eval_dn).readlines()
		print("Data Samples: ", len(txt_eval_dn))

	if not args.da == 'uda':
		label_map_s = {}
		for i in range(len(args.src_classes)):
			label_map_s[args.src_classes[i]] = i
		
		new_src = []
		for i in range(len(txt_src)):
			rec = txt_src[i]
			reci = rec.strip().split(' ')
			if int(reci[1]) in args.src_classes:
				line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
				new_src.append(line)
		txt_src = new_src.copy()

		new_tar = []
		for i in range(len(txt_test)):
			rec = txt_test[i]
			reci = rec.strip().split(' ')
			if int(reci[1]) in args.tar_classes:
				if int(reci[1]) in args.src_classes:
					line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
					new_tar.append(line)
				else:
					line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
					new_tar.append(line)
		txt_test = new_tar.copy()

	if args.trte == "val":
		dsize = len(txt_src)
		tr_size = int(0.9*dsize)
		# print(dsize, tr_size, dsize - tr_size)
		tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
	else:
		dsize = len(txt_src)
		tr_size = int(0.9*dsize)
		_, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
		tr_txt = txt_src

	dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
	dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers= args.worker, drop_last=False)
	dsets["source_te"] = ImageList(te_txt, transform=image_test())
	dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers= args.worker, drop_last=False)
	
	dsets["test"] = ImageList(txt_test, transform=image_test())
	dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers= args.worker, drop_last=False)

	if args.dset =='domain_net':
		dsets["eval_dn"] = ImageList_idx(txt_eval_dn, transform=image_train())
		dset_loaders["eval_dn"] = DataLoader(dsets["eval_dn"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
											drop_last=False)
	else:
		dset_loaders["eval_dn"] = dset_loaders["test"]

	return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
	start_test = True
	with torch.no_grad():
		iter_test = iter(loader)
		for i in tqdm(range(len(loader))):
			data = next(iter_test)
			inputs = data[0]
			labels = data[1]
			inputs = inputs.cuda()
			outputs = netC(netB(netF(inputs)))
			if start_test:
				all_output = outputs.float().cpu()
				all_label = labels.float()
				start_test = False
			else:
				all_output = torch.cat((all_output, outputs.float().cpu()), 0)
				all_label = torch.cat((all_label, labels.float()), 0)

	all_output = nn.Softmax(dim=1)(all_output)
	_, predict = torch.max(all_output, 1)
	accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
	mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
	if flag:
		matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
		acc = matrix.diagonal()/matrix.sum(axis=1) * 100
		aacc = acc.mean()
		aa = [str(np.round(i, 2)) for i in acc]
		acc = ' '.join(aa)
		return aacc, acc
	else:
		return accuracy*100, mean_ent

def cal_acc_oda(loader, netF, netB, netC):
	start_test = True
	with torch.no_grad():
		iter_test = iter(loader)
		for i in range(len(loader)):
			data = next(iter_test)
			inputs = data[0]
			labels = data[1]
			inputs = inputs.cuda()
			outputs = netC(netB(netF(inputs)))
			if start_test:
				all_output = outputs.float().cpu()
				all_label = labels.float()
				start_test = False
			else:
				all_output = torch.cat((all_output, outputs.float().cpu()), 0)
				all_label = torch.cat((all_label, labels.float()), 0)

	all_output = nn.Softmax(dim=1)(all_output)
	_, predict = torch.max(all_output, 1)
	ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
	ent = ent.float().cpu()
	initc = np.array([[0], [1]])
	kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
	threshold = (kmeans.cluster_centers_).mean()

	predict[ent>threshold] = args.class_num
	matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
	matrix = matrix[np.unique(all_label).astype(int),:]

	acc = matrix.diagonal()/matrix.sum(axis=1) * 100
	unknown_acc = acc[-1:].item()

	return np.mean(acc[:-1]), np.mean(acc), unknown_acc
	# return np.mean(acc), np.mean(acc[:-1])

def train_source(args):
	dset_loaders = data_load(args)
	## set base network
	if args.net[0:3] == 'res':
		netF = network.ResBase(res_name=args.net,se=args.se,nl=args.nl).cuda()
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

	
	### test model paremet size
	# model=network.ResBase(res_name=args.net)
	# num_params = sum([np.prod(p.size()) for p in model.parameters()])
	# print("Total number of parameters: {}".format(num_params))
	# 
	# num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
	# print("Total number of learning parameters: {}".format(num_params_update))

	netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
	netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

	if torch.cuda.device_count() >= 1:
		gpu_list = []
		for i in range(len(args.gpu_id.split(','))):
			gpu_list.append(i)
		print("Let's use", len(gpu_list), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		netF = nn.DataParallel(netF, device_ids=gpu_list)
		netB = nn.DataParallel(netB, device_ids=gpu_list)
		netC = nn.DataParallel(netC, device_ids=gpu_list)

	param_group = []
	learning_rate = args.lr
	for k, v in netF.named_parameters():
		param_group += [{'params': v, 'lr': learning_rate*0.1}]
	for k, v in netB.named_parameters():
		param_group += [{'params': v, 'lr': learning_rate}]
	for k, v in netC.named_parameters():
		param_group += [{'params': v, 'lr': learning_rate}]   
	optimizer = optim.SGD(param_group)
	optimizer = op_copy(optimizer)

	acc_init = 0
	max_iter = args.max_epoch * len(dset_loaders["source_tr"])
	interval_iter = max_iter // args.interval
	iter_num = 0

	netF.train()
	netB.train()
	netC.train()

	while iter_num < max_iter:
		try:
			inputs_source, labels_source = next(iter_source)
		except:
			iter_source = iter(dset_loaders["source_tr"])
			inputs_source, labels_source = next(iter_source)

		if inputs_source.size(0) == 1:
			continue


		inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
		outputs_source = netC(netB(netF(inputs_source)))
		#print(args.class_num, outputs_source.shape, labels_source.shape)
		
		
		classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
		wandb.log({'SRC Train: train_classifier_loss': classifier_loss.item()})

		optimizer.zero_grad()
		classifier_loss.backward()
		optimizer.step()

		if iter_num % interval_iter == 0 or iter_num == max_iter:
			lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

			netF.eval()
			netB.eval()
			netC.eval()
			if args.dset=='visda-2017':
				acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
				log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
			else:
				acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
				log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
			wandb.log({'SRC TRAIN: Acc' : acc_s_te})
			args.out_file.write(log_str + '\n')
			args.out_file.flush()
			print(log_str+'\n')

			if acc_s_te >= acc_init:

				acc_init = acc_s_te
				best_netF = netF.state_dict()
				best_netB = netB.state_dict()
				best_netC = netC.state_dict()

				torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
				torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
				torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
				print('Model Saved!!')

			netF.train()
			netB.train()
			netC.train()
		iter_num += 1
	# torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
	# torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
	# torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))
	# print('Final Model Saved!!')

	return netF, netB, netC

def test_target(args):
	dset_loaders = data_load(args)
	## set base network
	if args.net[0:3] == 'res':
		netF = network.ResBase(res_name=args.net,se=args.se,nl=args.nl).cuda()
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
		
	netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
	netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
	
	if torch.cuda.device_count() >= 1:
		gpu_list = []
		for i in range(len(args.gpu_id.split(','))):
			gpu_list.append(i)
		print("Let's use", len(gpu_list), "GPUs!")
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
		netF = nn.DataParallel(netF, device_ids=gpu_list)
		netB = nn.DataParallel(netB, device_ids=gpu_list)
		netC = nn.DataParallel(netC, device_ids=gpu_list)

	args.modelpath = args.output_dir_src + '/source_F.pt'   
	netF.load_state_dict(torch.load(args.modelpath))
	args.modelpath = args.output_dir_src + '/source_B.pt'   
	netB.load_state_dict(torch.load(args.modelpath))
	args.modelpath = args.output_dir_src + '/source_C.pt'   
	netC.load_state_dict(torch.load(args.modelpath))
	netF.eval()
	netB.eval()
	netC.eval()

	if args.da == 'oda':
		acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
		log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
	else:
		if args.dset=='visda-2017':
			acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
			log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
		else:
			acc, _ = cal_acc(dset_loaders['eval_dn'], netF, netB, netC, False)
			log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

	args.out_file.write(log_str)
	args.out_file.flush()
	print(log_str)

def print_args(args):
	s = "==========================================\n"
	for arg, content in args.__dict__.items():
		s += "{}:{}\n".format(arg, content)
	return s

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='SHOT')
	parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
	parser.add_argument('--s', type=int, default=0, help="source")
	parser.add_argument('--t', type=int, default=1, help="target")
	parser.add_argument('--max_epoch', type=int, default=200, help="max iterations")
	parser.add_argument('--interval', type=int, default=50, help="interval")
	parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
	parser.add_argument('--dset', type=str, default='office-home', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])
	parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
	parser.add_argument('--net', type=str, default='deit_s', help="vgg16, resnet50, resnet101, vit, deit_s")
	parser.add_argument('--seed', type=int, default=2020, help="random seed")
	parser.add_argument('--bottleneck', type=int, default=256)
	parser.add_argument('--epsilon', type=float, default=1e-5)
	parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
	parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
	parser.add_argument('--smooth', type=float, default=0.1)   
	parser.add_argument('--output', type=str, default='src_train')
	parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
	parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
	parser.add_argument('--bsp', type=bool, default=False)
	parser.add_argument('--se', type=bool, default=False)
	parser.add_argument('--nl', type=bool, default=False)
	parser.add_argument('--worker', type=int, default=16)
	parser.add_argument('--wandb', type=int, default=0)

	args = parser.parse_args()

	if args.dset == 'office-home':
		names = ['Art', 'Clipart', 'Product', 'RealWorld']
		args.class_num = 65 
	if args.dset == 'office':
		names = ['amazon', 'dslr', 'webcam']
		args.class_num = 31
	if args.dset == 'visda-2017':
		names = ['train', 'validation']
		args.class_num = 12
	if args.dset == 'office-caltech':
		names = ['amazon', 'caltech', 'dslr', 'webcam']
		args.class_num = 10
	if args.dset == 'pacs':
		names = ['art_painting', 'cartoon', 'photo', 'sketch']
		args.class_num = 7
	if args.dset =='domain_net':
		names = ['clipart', 'infograph', 'painting', 'quickdraw', 'sketch', 'real']
		args.class_num = 345
	if args.dset =='pacs':
		names = ['art_painting','cartoon', 'photo', 'sketch']
		args.class_num = 7


	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
	SEED = args.seed
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	np.random.seed(SEED)
	random.seed(SEED)
	# torch.backends.cudnn.deterministic = True

	folder = './data/'
	args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
	args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'    
	
	mode = 'online' if args.wandb else 'disabled'
	wandb.init(project='CoNMix ECCV', name=f'SRC {names[args.s]}', mode=mode, config=args, tags=['SRC', args.dset, args.net])

	print(print_args(args))
	if args.dset == 'office-home':
		if args.da == 'pda':
			args.class_num = 65
			args.src_classes = [i for i in range(65)]
			args.tar_classes = [i for i in range(25)]
		if args.da == 'oda':
			args.class_num = 25
			args.src_classes = [i for i in range(25)]
			args.tar_classes = [i for i in range(65)]

	args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
	args.name_src = names[args.s][0].upper()
	if not osp.exists(args.output_dir_src):
		os.system('mkdir -p ' + args.output_dir_src)
	if not osp.exists(args.output_dir_src):
		os.mkdir(args.output_dir_src)

	args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
	args.out_file.write(print_args(args)+'\n')
	args.out_file.flush()
	train_source(args)

	args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
	for i in range(len(names)):
		if i == args.s:
			continue
		args.t = i
		args.name = names[args.s][0].upper() + names[args.t][0].upper()

		folder = './data/'
		args.s_dset_path = folder + args.dset + '/' + names[args.s] + '.txt'
		args.test_dset_path = folder + args.dset + '/' + names[args.t] + '.txt'
		if args.dset =='domain_net':
			args.txt_eval_dn = folder + args.dset + '/' + names[args.t] + '_test.txt'
		else:
			args.txt_eval_dn = args.test_dset_path


		if args.dset == 'office-home':
			if args.da == 'pda':
				args.class_num = 65
				args.src_classes = [i for i in range(65)]
				args.tar_classes = [i for i in range(25)]
			if args.da == 'oda':
				args.class_num = 25
				args.src_classes = [i for i in range(25)]
				args.tar_classes = [i for i in range(65)]

		test_target(args)