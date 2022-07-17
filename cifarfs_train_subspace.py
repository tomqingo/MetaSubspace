import  torch, os
import  numpy as np
from cifarfs import CIFAR100_FS
import  argparse

from    meta_psgd import Meta
import pdb
import logging
import myutils as mtils
from PIL import Image
import cv2
import pickle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.io as io

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    mtils.setup_logging(os.path.join(args.log_dir,'log.txt'))
    logging.info("saving to %s", os.path.join(args.log_dir,'log.txt'))
    logging.debug("run arguments: %s", args)    

    #config_net = [
    #    ('conv2d', [96, 3, 3, 3, 1, 1]),
    #    ('relu', [True]),
    #    ('bn', [32]),
    #    ('conv2d', [32, 32, 3, 3, 2, 0]),
    #    ('relu', [True]),
    #    ('bn', [32]),
    #    ('conv2d', [32, 32, 3, 3, 2, 0]),
    #    ('relu', [True]),
    #    ('bn', [32]),
    #    ('conv2d', [32, 32, 2, 2, 1, 0]),
    #    ('relu', [True]),
    #    ('bn', [32]),
    #    ('flatten', []),
    #    ('linear', [args.n_way, 128])
    #]

    config_net = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 2 * 2])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config_net).to(device)
    #pdb.set_trace()

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    logging.info('Num of training parameters:{0}'.format(num))

    db_train = CIFAR100_FS('/home/datasets/CIFAR100',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
    
    best_prec1_test = 0.0

    train_transform = transforms.Compose([
    transforms.RandomCrop(args.imgsz, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    
    epoch_col = []
    losses_orth_col = []
    losses_classify_col = []

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()

        #for task_id in range(y_spt.shape[0]):
        #    label_col = [0,0,0,0,0]
        #    if not os.path.exists(os.path.join(args.log_dir, 'image', str(task_id))):
        #            os.makedirs(os.path.join(args.log_dir, 'image', str(task_id)))
        #    orgpath = os.path.join(args.log_dir, 'image', str(task_id))
        #    for img_id in range(y_spt.shape[1]):
        #        img = x_spt[task_id,img_id]
        #        label = y_spt[task_id,img_id]
                #pdb.set_trace()
        #        cv2.imwrite(os.path.join(orgpath, str(label)+'_'+str(label_col[label])+'.png'),img)
        #        label_col[label] = label_col[label] + 1
        #pdb.set_trace()
        #pdb.set_trace()
        x_spt = x_spt.reshape(-1, args.imgsz, args.imgsz, args.imgc)
        x_qry = x_qry.reshape(-1, args.imgsz, args.imgsz, args.imgc)

        x_spt_tensor = Image.fromarray(np.uint8(x_spt[0]))
        x_spt_tensor = train_transform(x_spt_tensor)
        # save the original image


        for x_spt_id in range(1,x_spt.shape[0]):
            x_spt_per = Image.fromarray(np.uint8(x_spt[x_spt_id]))
            x_spt_per = train_transform(x_spt_per)
            #pdb.set_trace()
            x_spt_tensor = torch.cat([x_spt_tensor, x_spt_per], dim=0)

        x_spt_tensor = x_spt_tensor.reshape(args.task_num,args.n_way*args.k_spt,args.imgc,args.imgsz,args.imgsz)

        x_qry_tensor = Image.fromarray(np.uint8(x_qry[0]))
        x_qry_tensor = test_transform(x_qry_tensor)

        for x_qry_id in range(1,x_qry.shape[0]):
            x_qry_per = Image.fromarray(np.uint8(x_qry[x_qry_id]))
            x_qry_per = test_transform(x_qry_per)
            x_qry_tensor = torch.cat([x_qry_tensor, x_qry_per], dim=0)
        
        x_qry_tensor = x_qry_tensor.reshape(args.task_num,args.n_way*args.k_qry,args.imgc,args.imgsz,args.imgsz)
        
        x_spt_tensor, y_spt, x_qry_tensor, y_qry = x_spt_tensor.to(device), torch.from_numpy(y_spt).to(device), \
                                     x_qry_tensor.to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        loss_classify, loss_orth, accs = maml(x_spt_tensor, y_spt, x_qry_tensor, y_qry)

        if step % 1 == 0:
            print('step:', step, '\ttraining acc:', accs)
            logging.info('step:{0}, training acc:{1}'.format(str(step), " ".join(str(acc) for acc in accs)))

        if step % 50 == 0:
            epoch_col.append(step)
            losses_orth_col.append(loss_orth.detach().data.cpu().numpy().tolist())
            losses_classify_col.append(loss_classify.detach().data.cpu().numpy().tolist())
            print(epoch_col, losses_orth_col, losses_classify_col)
            accs = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')

                #for task_id in range(y_spt.shape[0]):
                #    label_col = [0,0,0,0,0]
                #    if not os.path.exists(os.path.join(args.log_dir, 'image', str(task_id))):
                #        os.makedirs(os.path.join(args.log_dir, 'image', str(task_id)))
                #        orgpath = os.path.join(args.log_dir, 'image', str(task_id))
                #        for img_id in range(y_spt.shape[1]):
                #            img = x_spt[task_id,img_id]
                #            label = y_spt[task_id,img_id]
                            #pdb.set_trace()
                #            cv2.imwrite(os.path.join(orgpath, str(label)+'_'+str(label_col[label])+'.png'),img)
                #            label_col[label] = label_col[label] + 1
                #pdb.set_trace()               
                x_spt = x_spt.reshape(-1, args.imgsz, args.imgsz, args.imgc)
                x_qry = x_qry.reshape(-1, args.imgsz, args.imgsz, args.imgc)

                # pdb.set_trace()
                x_spt_tensor = Image.fromarray(np.uint8(x_spt[0]))
                x_spt_tensor = train_transform(x_spt_tensor)
                #pdb.set_trace()

                for x_spt_id in range(1,x_spt.shape[0]):
                    x_spt_per = Image.fromarray(np.uint8(x_spt[x_spt_id]))
                    x_spt_per = train_transform(x_spt_per)
                    x_spt_tensor = torch.cat([x_spt_tensor, x_spt_per], dim=0)

                x_spt_tensor = x_spt_tensor.reshape(args.task_num,args.n_way*args.k_spt,args.imgc,args.imgsz,args.imgsz)

                x_qry_tensor = Image.fromarray(np.uint8(x_qry[0]))
                x_qry_tensor = test_transform(x_qry_tensor)

                for x_qry_id in range(1,x_qry.shape[0]):
                    x_qry_per = Image.fromarray(np.uint8(x_qry[x_qry_id]))
                    x_qry_per = test_transform(x_qry_per)
                    x_qry_tensor = torch.cat([x_qry_tensor, x_qry_per], dim=0)
        
                x_qry_tensor = x_qry_tensor.reshape(args.task_num,args.n_way*args.k_qry,args.imgc,args.imgsz,args.imgsz)
        
                x_spt_tensor, y_spt, x_qry_tensor, y_qry = x_spt_tensor.to(device), torch.from_numpy(y_spt).to(device), \
                                     x_qry_tensor.to(device), torch.from_numpy(y_qry).to(device)


                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt_tensor, y_spt, x_qry_tensor, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            #pdb.set_trace()
            accs_mean = np.array(accs).mean(axis=0).astype(np.float16)
            accs_max = np.array(accs).max(axis=1).astype(np.float16)
            accs_std = np.std(accs_max)

            print('Test acc:', accs_mean)
            print('Test acc one:', np.max(accs_mean))

            is_best_test = max(accs_mean)>best_prec1_test
            if is_best_test:
                best_test_epoch = step+1
                projection_variable_save = getattr(maml, 'projection_variable').data
                projection_variable_save = projection_variable_save.cpu().numpy()
            best_prec1_test = max(max(accs_mean),best_prec1_test)
            print('Best test acc one:', best_prec1_test)
            print('Test acc std:', accs_std)

            mtils.save_checkpoint({
            'epoch': step + 1,
            'model': 'Meta',
            'state_dict':  maml.state_dict(),
            'best_prec1_test': best_prec1_test, 
            'best_test_epoch': best_test_epoch
            }, is_best=is_best_test, path=args.log_dir)

            logging.info('step:{0}, testing acc:{1}'.format(str(step), " ".join(str(acc) for acc in accs_mean)))
            logging.info('step:{0}, best testing acc:{1}, acc_std:{2}'.format(str(step), str(best_prec1_test), str(accs_std)))
        
        with open(os.path.join(args.log_dir, 'subspace_best.txt'), 'wb') as file:
            pickle.dump(projection_variable_save, file)
        
        io.savemat(os.path.join(args.log_dir, 'loss.mat'), {'classloss':losses_classify_col, 'orthloss': losses_orth_col})
        



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--num_subspace', type=int, help='number of subspace base vectors', default=40)
    argparser.add_argument('--log_dir', type=str, help='the log file directory', default='./save_final/cifarfs/subspace/5_way_5_shot')

    args = argparser.parse_args()

    main(args)
