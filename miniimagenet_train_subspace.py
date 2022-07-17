import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import myutils as mtils

#from meta import Meta
from meta_psgd import Meta
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    mtils.setup_logging(os.path.join(args.log_dir,'log.txt'))
    logging.info("saving to %s", os.path.join(args.log_dir,'log.txt'))
    logging.debug("run arguments: %s", args) 

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)



    # batchsz here means total episode number
    mini = MiniImagenet('/home/datasets/mini-imagenet/mini-imagenet', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=1600, resize=args.imgsz)
    mini_test = MiniImagenet('/home/datasets/mini-imagenet/mini-imagenet', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=16, resize=args.imgsz)

    best_prec1_test = 0.0

    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=4, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 1 == 0:
                print('step:', step, '\ttraining acc:', accs)
                logging.info('step:{0}, training acc:{1}'.format(str(step), " ".join(str(acc) for acc in accs)))

            if step % 50 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs_mean = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accs_max = np.array(accs_all_test).max(axis=1).astype(np.float16)
                accs_std = np.std(accs_max)

                print('Test acc:', accs_mean)
                print('Test acc one:', np.max(accs_mean))

                is_best_test = max(accs_mean)>best_prec1_test
                if is_best_test:
                    best_test_epoch = step+1
                best_prec1_test = max(max(accs_mean),best_prec1_test)
                print('Best test acc one:', best_prec1_test, 'Test acc std:', accs_std)

                mtils.save_checkpoint({
                'epoch': step + 1,
                'model': 'Meta',
                'state_dict':  maml.state_dict(),
                'best_prec1_test': best_prec1_test, 
                'best_test_epoch': best_test_epoch
                }, is_best=is_best_test, path=args.log_dir)

                logging.info('step:{0}, testing acc:{1}'.format(str(step), " ".join(str(acc) for acc in accs_mean)))
                logging.info('step:{0}, best testing acc:{1}, acc_std:{2}'.format(str(step), str(best_prec1_test), str(accs_std)))

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--num_subspace', type=int, help='number of subspace base vectors', default=40)
    argparser.add_argument('--log_dir', type=str, help='the log file directory', default='./save_dir/miniimagenet/subspace/lab_0')
    args = argparser.parse_args()

    main()
