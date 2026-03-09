from ntpath import join
import os
from torch.autograd import Variable
import torch

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


def get_variable(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape, f"Predicted shape is {predicted.shape} while target is {target.shape}"
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def pck_cal(predicted, target):
    assert predicted.shape == target.shape
    threshold = 150

    frame_num = predicted.shape[1]*1.0
    joints_num = predicted.shape[-2]*1.0
    
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)

    t = torch.Tensor([threshold]).cuda()
    out = (dis < t).float() * 1
    pck = out.sum()/ joints_num / frame_num

    return pck


def auc_cal(predicted, target):
    assert predicted.shape == target.shape
    dis = torch.norm(predicted - target, dim=len(target.shape)-1)
    outall = 0
    threshold = 150

    frame_num = predicted.shape[1]*1.0
    joints_num = predicted.shape[-2]*1.0

    for i in range(threshold):
        t = torch.Tensor([float(i)]).cuda()
        out = (dis < t).float() * 1
        outall+=out.sum() /joints_num / frame_num

    outall = outall/threshold
    
    return outall



def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_p1, wandb_id, last=True):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    file_name = 'last.pth.tr' if last else 'best.pth.tr'
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_p1': min_p1,
        'wandb_id': wandb_id,
    }, os.path.join(checkpoint_path, file_name))