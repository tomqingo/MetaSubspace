import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
from    torch.autograd import Variable
import  numpy as np


from    learner import Learner
from    copy import deepcopy
import pdb
from sklearn.decomposition import PCA


def get_model_param_vec(model):

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.detach().reshape(-1))
    return torch.cat(vec,0)

def get_model_grad_vec(grad):

    vec = []
    for idx in range(len(grad)):
        vec.append(grad[idx].reshape(-1))
    return torch.cat(vec,0)

def update_grad(model, grad_vec):

    idx = 0
    grad = []
    #pdb.set_trace()
    for name, param in model.named_parameters():
        arr_shape = param.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        grad.append(grad_vec[idx:idx+size].reshape(arr_shape))
        idx += size
        if idx==grad_vec.shape[0]:
            break
    grad = tuple(grad)
    return grad

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        #self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # generate a random orthogonal basis

        param_num = sum(x.numel() for x in self.net.parameters() if x.requires_grad)
        projection_large = np.random.randn(50,param_num)
        pca = PCA(n_components=args.num_subspace)
        pca.fit_transform(projection_large)
        projection_matrix = np.array(pca.components_)
        #Q, R = np.linalg.qr(projection_large)
        #projection_matrix = Q[param_num, 0:41]
        device = torch.device('cuda')
        self.projection_tensor = torch.Tensor(projection_matrix)
        #self.projection_variable_new = Variable(self.projection_tensor, requires_grad=True)
        #self.projection_variable = self.projection_variable_new.to(device)
        self.projection_variable = torch.nn.Parameter(self.projection_tensor)
        self.meta_optim = optim.Adam([self.projection_variable], lr=self.meta_lr)
        self.meta_param_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
 



    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            grad_vec = get_model_grad_vec(grad)
            #pdb.set_trace()
            gk = torch.mm(self.projection_variable, grad_vec.reshape(-1,1))
            grad_proj = torch.mm(self.projection_variable.transpose(1,0), gk)
            grad_new = update_grad(self.net, grad_proj)

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_new, self.net.parameters())))


            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                grad_vec = get_model_grad_vec(grad)
                gk = torch.mm(self.projection_variable, grad_vec.reshape(-1,1))
                grad_proj = torch.mm(self.projection_variable.transpose(1,0), gk)
                grad_new = update_grad(self.net, grad_proj)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_new, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] = losses_q[k+1] + loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_classify = losses_q[-1] / task_num
        #loss_orth = torch.sqrt(torch.sum((torch.mm(self.projection_variable,self.projection_variable.transpose(1,0))-torch.eye(40).cuda())**2))
        #loss_orth = torch.norm(torch.mm(self.projection_variable,self.projection_variable.transpose(1,0))-torch.eye(40).cuda(), p='fro')
        proj_n = 40
        multi_matrix = torch.mm(self.projection_variable,self.projection_variable.transpose(1,0))
        multi_matrix = multi_matrix.flatten()[:-1].view(proj_n-1,proj_n+1)[:,1:].flatten()
        loss_orth = torch.sqrt(torch.sum(multi_matrix**2))
        loss_all = loss_classify + 1e-5*loss_orth

        # optimize theta parameters
        self.meta_optim.zero_grad()
        #self.meta_param_optim.zero_grad()
        loss_all.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        #print(self.projection_variable.grad)
        #pdb.set_trace()
        #self.meta_param_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return loss_classify, loss_orth, accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        grad_vec = get_model_grad_vec(grad)
        gk = torch.mm(self.projection_variable, grad_vec.reshape(-1,1))
        grad_proj = torch.mm(self.projection_variable.transpose(1,0), gk)
        grad_new = update_grad(self.net, grad_proj)

        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_new, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            grad_vec = get_model_grad_vec(grad)
            gk = torch.mm(self.projection_variable, grad_vec.reshape(-1,1))
            grad_proj = torch.mm(self.projection_variable.transpose(1,0), gk)
            grad_new = update_grad(self.net, grad_proj)

            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_new, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
