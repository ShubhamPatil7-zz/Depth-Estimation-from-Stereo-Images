from __future__ import print_function
import argparse
import time
import numpy as np
import skimage
import skimage.io
import skimage.transform
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from models import *
from utils import preprocess

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2012',
                    help='KITTI version')
parser.add_argument('--datapath', default='./data/KITTI2012/training/',
                    help='select model')
parser.add_argument('--loadmodel', default='./saved_models/pretrained_model_KITTI2012.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

cuda_device_id = 7

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(cuda_device_id)

if args.KITTI == '2015':
    from dataloader import KITTIloader2015 as DA
else:
    from dataloader import KITTIloader2012 as DA

test_left_img, test_right_img, disp_train_L, left_val, right_val, disp_val_L = DA.dataloader(args.datapath)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

# model = nn.DataParallel(model)
model.cuda()
if args.loadmodel is not None:
    from collections import OrderedDict

    state_dict = torch.load(args.loadmodel)
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def test(imgL, imgR):
    model.eval()
    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda()
    imgL, imgR = Variable(imgL), Variable(imgR)
    with torch.no_grad():
        output = model(imgL, imgR)
    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()
    return pred_disp


def main():
    processed = preprocess.get_transform(augment=False)
    for inx in range(len(test_left_img)):
        imgL_o = (skimage.io.imread(test_left_img[inx]).astype('float32'))
        imgR_o = (skimage.io.imread(test_right_img[inx]).astype('float32'))
        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])
        imgR = np.reshape(imgR, [1, 3, imgR.shape[1], imgR.shape[2]])

        # pad to (384, 1248)
        top_pad = 384 - imgL.shape[2]
        left_pad = 1248 - imgL.shape[3]
        imgL = np.lib.pad(imgL, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)
        imgR = np.lib.pad(imgR, ((0, 0), (0, 0), (top_pad, 0), (0, left_pad)), mode='constant', constant_values=0)

        start_time = time.time()
        pred_disp = test(imgL, imgR)
        print('time = %.2f' % (time.time() - start_time))

        top_pad = 384 - imgL_o.shape[0]
        left_pad = 1248 - imgL_o.shape[1]
        img = pred_disp[top_pad:, :-left_pad]
        skimage.io.imsave('ouput/KITTI2012_Train/' + test_left_img[inx].split('/')[-1], (img * 256).astype('uint16'))


if __name__ == '__main__':
    main()
