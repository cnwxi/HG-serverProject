import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn, optim
from feeders.feeder import Feeder
from model.msg3d import Model
from torch.optim.lr_scheduler import MultiStepLR

num_joint = 18
max_frame = 300
num_person_out = 2
num_person_in = 5

dataPath = '../data/my_val'
labelPath = '../data/my_val_label.json'
saveJoint = '../data/testJoint.npy'
saveBone = '../data/testBone.npy'
saveLabel = '../data/testLabel.pkl'


# self.load_model()
# self.load_param_groups()
# self.load_optimizer()
# self.load_lr_scheduler()
# self.load_data()
class MSG3D:
    def __init__(self,
                 num_class=10,
                 num_point=18,
                 num_person=2,
                 num_gcn_scales=8,
                 num_g3d_scales=8,
                 graph='graph.kinetics.AdjMatrixGraph',
                 device=0,
                 # weights=None,
                 weight_decay=0.003,
                 base_lr=0.1,
                 step=[45, 45],
                 nesterov=True):
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.num_gcn_scales = num_gcn_scales
        self.num_g3d_scales = num_g3d_scales
        self.graph = graph
        self.device = device
        # self.weights = weights
        self.weights_decay = weight_decay
        self.lr = base_lr
        self.step = step
        self.nesterov = nesterov

    def load(self):
        self.model = Model(self.num_class,
                           self.num_point,
                           self.num_person,
                           self.num_gcn_scales,
                           self.num_g3d_scales,
                           self.graph).cuda()
        self.lose = nn.CrossEntropyLoss().cuda(self.device)
        # self.model.load_state_dict(self.weights)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weights_decay,
                                   nesterov=self.nesterov)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.step, gamma=0.1)

    def load_data(self, data_path, label_path, batch_size):
        def workfn(worker_id):
            seed = worker_id + random.randrange(200) + 1
            torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            return

        self.data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=data_path, label_path=label_path),
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            worker_init_fn=workfn
        )

    def tarin(self,epoch, forward_batch_size):
        self.model.train()
        loader = self.data_loader
        loss_values = []

        for batch_idx, (data, label, index) in enumerate(loader):
            with torch.no_grad():
                data = data.float().cuda(self.device)
                label = label.long().cuda(self.device)
            self.optimizer.zero_grad()
            splits = len(data) // forward_batch_size
            assert len(data) % forward_batch_size == 0, \
                '实际的批量大小应该是批量大小的因子'
            for i in range(splits):
                left = i * forward_batch_size
                right = left + forward_batch_size
                batch_data, batch_label = data[left:right], label[left:right]
                output = self.model(batch_data)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.lose(output, batch_label) / splits
                loss.backward()
                loss_values.append(loss.item())
                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())
                print('acc', acc)
                print('loss', loss.item() * splits)
                print('loss_l1', l1)
            self.optimizer.step()
            del output
            del loss

        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])
        save_path = f'../msg3dModels/myModel_{epoch}.pt'
        torch.save(weights, save_path)


if __name__ == '__main__':
    batch_size = 6
    forward_batch_size = 6
    msg3d = MSG3D()
    msg3d.load()
    msg3d.load_data(data_path='../data/testJoint.npy',
                    label_path='../data/testLabel.pkl',
                    batch_size=batch_size)
    epoch = 1
    for i in epoch:
        msg3d.tarin(epoch,forward_batch_size)
