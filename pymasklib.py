from __future__ import print_function, division, absolute_import, unicode_literals

#import required modules
print('importing...')
#from google.colab import files
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image as Img
import cv2
from skimage.transform import resize
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

# data loader

from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
print('Done!')


# In[2]:


from skimage import io


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


os.getcwd()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:






import codecs
import yaml
import six
import time

from ast import literal_eval

class SegmentronConfig(dict):
    def __init__(self, *args, **kwargs):
        super(SegmentronConfig, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        if key in ["immutable"]:
            if key not in self.__dict__:
                self.__dict__[key] = False
            return self.__dict__[key]

        if not key in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = SegmentronConfig()
        return self[key]

    def __setitem__(self, key, value):
        #
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, six.string_types):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(SegmentronConfig, self).__setitem__(key, value)

    def update_from_other_cfg(self, other):
        if isinstance(other, dict):
            other = SegmentronConfig(other)
        assert isinstance(other, SegmentronConfig)
        cfg_list = [("", other)]
        while len(cfg_list):
            prefix, tdic = cfg_list[0]
            cfg_list = cfg_list[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    cfg_list.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def remove_irrelevant_cfg(self):
        model_name = self.MODEL.MODEL_NAME

        from ..models.model_zoo import MODEL_REGISTRY
        model_list = MODEL_REGISTRY.get_list()
        model_list_lower = [x.lower() for x in model_list]
        # print('model_list:', model_list)
        assert model_name.lower() in model_list_lower, "Expected model name in {}, but received {}"            .format(model_list, model_name)
        pop_keys = []
        for key in self.MODEL.keys():
            if key.lower() in model_list_lower and key.lower() != model_name.lower():
                pop_keys.append(key)
        for key in pop_keys:
            self.MODEL.pop(key)



    def check_and_freeze(self):
        self.TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        # TODO: remove irrelevant config and then freeze
        self.remove_irrelevant_cfg()
        self.immutable = True

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file="translab.yaml"):
        #with codecs.open(config_file, 'r', 'utf-8') as file:
        #    loaded_cfg = yaml.load(file, Loader=yaml.FullLoader)
        loaded_cfg = {'DATASET': {'NAME': 'trans10k_boundary', 'MEAN': [0.485, 0.456, 0.406], 'STD': [0.229, 0.224, 0.225]}, 'TRAIN': {'EPOCHS': 16, 'BATCH_SIZE': 8, 'CROP_SIZE': 769, 'MODEL_SAVE_DIR': 'workdirs/debug'}, 'TEST': {'BATCH_SIZE': 1}, 'SOLVER': {'LR': 0.02}, 'MODEL': {'MODEL_NAME': 'TransLab', 'BACKBONE': 'resnet50'}}
            #print(f'loaded_cfg: {loaded_cfg}')
        self.update_from_other_cfg(loaded_cfg)

    def set_immutable(self, immutable):
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, SegmentronConfig):
                value.set_immutable(immutable)

    def is_immutable(self):
        return self.immutable



#get_ipython().system('pwd')

# from segmentron.config import cfg

cfg = SegmentronConfig()

########################## basic set ###########################################
# random seed
cfg.SEED = 1024
# train time stamp, auto generate, do not need to set
cfg.TIME_STAMP = ''
# root path
cfg.ROOT_PATH = ''
# model phase ['train', 'test']
cfg.PHASE = 'test'

########################## dataset config #########################################
# dataset name
cfg.DATASET.NAME = 'trans10k_extra'
# pixel mean
cfg.DATASET.MEAN = [0.5, 0.5, 0.5]
# pixel std
cfg.DATASET.STD = [0.5, 0.5, 0.5]
# dataset ignore index
cfg.DATASET.IGNORE_INDEX = -1
# workers
cfg.DATASET.WORKERS = 8
# val dataset mode
cfg.DATASET.MODE = 'testval'
########################### data augment ######################################
# data augment image mirror
cfg.AUG.MIRROR = True
# blur probability
cfg.AUG.BLUR_PROB = 0.0
# blur radius
cfg.AUG.BLUR_RADIUS = 0.0
# color jitter, float or tuple: (0.1, 0.2, 0.3, 0.4)
cfg.AUG.COLOR_JITTER = None
########################### train config ##########################################
# epochs
cfg.TRAIN.EPOCHS = 30
# batch size
cfg.TRAIN.BATCH_SIZE = 1
# train crop size
cfg.TRAIN.CROP_SIZE = 769
# train base size
cfg.TRAIN.BASE_SIZE = 512
# model output dir
cfg.TRAIN.MODEL_SAVE_DIR = 'workdirs/'
# log dir
cfg.TRAIN.LOG_SAVE_DIR = cfg.TRAIN.MODEL_SAVE_DIR
# pretrained model for eval or finetune
cfg.TRAIN.PRETRAINED_MODEL_PATH = ''
# use pretrained backbone model over imagenet
cfg.TRAIN.BACKBONE_PRETRAINED = True
# backbone pretrained model path, if not specific, will load from url when backbone pretrained enabled
cfg.TRAIN.BACKBONE_PRETRAINED_PATH = ''
# resume model path
cfg.TRAIN.RESUME_MODEL_PATH = ''
# whether to use synchronize bn
cfg.TRAIN.SYNC_BATCH_NORM = True
# save model every checkpoint-epoch
cfg.TRAIN.SNAPSHOT_EPOCH = 1

########################### optimizer config ##################################
# base learning rate
cfg.SOLVER.LR = 1e-4
# optimizer method
cfg.SOLVER.OPTIMIZER = "sgd"
# optimizer epsilon
cfg.SOLVER.EPSILON = 1e-8
# optimizer momentum
cfg.SOLVER.MOMENTUM = 0.9
# weight decay
cfg.SOLVER.WEIGHT_DECAY = 1e-4 #0.00004
# decoder lr x10
cfg.SOLVER.DECODER_LR_FACTOR = 10.0
# lr scheduler mode
cfg.SOLVER.LR_SCHEDULER = "poly"
# poly power
cfg.SOLVER.POLY.POWER = 0.9
# step gamma
cfg.SOLVER.STEP.GAMMA = 0.1
# milestone of step lr scheduler
cfg.SOLVER.STEP.DECAY_EPOCH = [10, 20]
# warm up epochs can be float
cfg.SOLVER.WARMUP.EPOCHS = 0.
# warm up factor
cfg.SOLVER.WARMUP.FACTOR = 1.0 / 3
# warm up method
cfg.SOLVER.WARMUP.METHOD = 'linear'
# whether to use ohem
cfg.SOLVER.OHEM = False
# whether to use aux loss
cfg.SOLVER.AUX = False
# aux loss weight
cfg.SOLVER.AUX_WEIGHT = 0.4
# loss name
cfg.SOLVER.LOSS_NAME = ''
########################## test config ###########################################
# val/test model path
cfg.TEST.TEST_MODEL_PATH = '16.pth'
# test batch size
cfg.TEST.BATCH_SIZE = 1
# eval crop size
cfg.TEST.CROP_SIZE = None
# multiscale eval
cfg.TEST.SCALES = [1.0]
# flip
cfg.TEST.FLIP = False

########################## visual config ###########################################
# visual result output dir
cfg.VISUAL.OUTPUT_DIR = '../runs/visual/'

########################## model #######################################
# model name
cfg.MODEL.MODEL_NAME = 'TransLab'
# model backbone
cfg.MODEL.BACKBONE = ''
# model backbone channel scale
cfg.MODEL.BACKBONE_SCALE = 1.0
# support resnet b, c. b is standard resnet in pytorch official repo
# cfg.MODEL.RESNET_VARIANT = 'b'
# multi branch loss weight
cfg.MODEL.MULTI_LOSS_WEIGHT = [1.0]
# gn groups
cfg.MODEL.DEFAULT_GROUP_NUMBER = 32
# whole model default epsilon
cfg.MODEL.DEFAULT_EPSILON = 1e-5
# batch norm, support ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
cfg.MODEL.BN_TYPE = 'BN'
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_ENCODER = None
# batch norm epsilon for encoder, if set None will use api default value.
cfg.MODEL.BN_EPS_FOR_DECODER = None
# backbone output stride
cfg.MODEL.OUTPUT_STRIDE = 16
# BatchNorm momentum, if set None will use api default value.
cfg.MODEL.BN_MOMENTUM = None


########################## DeepLab config ####################################
# whether to use aspp
cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP = True
# whether to use decoder
cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER = True
# whether aspp use sep conv
cfg.MODEL.DEEPLABV3_PLUS.ASPP_WITH_SEP_CONV = True
# whether decoder use sep conv
cfg.MODEL.DEEPLABV3_PLUS.DECODER_USE_SEP_CONV = True
########################## Demo ####################################

cfg.DEMO_DIR = ''



cfg

# from segmentron.utils.options import parse_args

import argparse

def parse_args():
    #print('start')
    
    parser = argparse.ArgumentParser(description='Segmentron')
    parser.add_argument('-f')
    parser.add_argument('--config-file', default='translab.yaml', metavar="FILE",
                        help='config file path')
    #print('1')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    #print('2')
    parser.add_argument('--local_rank', type=int, default=0)
    #print('3')
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    #print('4')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    #print('5')
    # for evaluation
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    #print('6')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    #print('7')
    # for visual
    parser.add_argument('--input-img', type=str, default='tools/demo_vis.png',
                        help='path to the input image or a directory of images')
    #print('8')
    # config options
    parser.add_argument('opts', help='See config for all options',
                        default=None, nargs=argparse.REMAINDER)
    #print('end')
    args = parser.parse_args()

    return args

#from segmentron.utils.default_setup import default_setup
"""
code is heavily based on https://github.com/facebookresearch/maskrcnn-benchmark
"""
import math
import pickle
import torch
import torch.utils.data as data
import torch.distributed as dist
import logging
import os
import sys
import logging
import numpy as np
import random
from datetime import datetime
import json


from torch.utils.data.sampler import Sampler, BatchSampler

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    

def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    if distributed_rank > 0:
        return

    logging.root.name = name
    logging.root.setLevel(logging.INFO)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logging.root.addHandler(fh)
        


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    
    
    
def default_setup(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if not args.no_cuda and torch.cuda.is_available():
        # cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO
    # if args.save_pred:
    #     outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)

    save_dir = cfg.TRAIN.MODEL_SAVE_DIR if cfg.PHASE == 'train' else None
    setup_logger("Segmentron", save_dir, get_rank(), filename='{}_{}_{}_{}_log.txt'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP))

    logging.info("Using {} GPUs".format(num_gpus))
    logging.info(args)
    logging.info(json.dumps(cfg, indent=8))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())

# from segmentron.utils.registry import Registry

# this code heavily based on detectron2

import logging
import torch

#from ..config import cfg

class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.

    To create a registry (inside segmentron):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name

        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)



    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))

        return ret

    def get_list(self):
        return list(self._obj_map.keys())



import logging
import torch

from collections import OrderedDict
#from segmentron.utils.registry import Registry
#from ..config import cfg

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for segment model, i.e. the whole model.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""


def get_segmentation_model():
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    model_name = cfg.MODEL.MODEL_NAME
    #print(model_name)
    #print(MODEL_REGISTRY.get_list())
    model = MODEL_REGISTRY.get(model_name)()
    load_model_pretrain(model)
    return model


def load_model_pretrain(model):
    if cfg.PHASE == 'train':
        if cfg.TRAIN.PRETRAINED_MODEL_PATH:
            logging.info('load pretrained model from {}'.format(cfg.TRAIN.PRETRAINED_MODEL_PATH))
            state_dict_to_load = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
            keys_wrong_shape = []
            state_dict_suitable = OrderedDict()
            state_dict = model.state_dict()
            for k, v in state_dict_to_load.items():
                if v.shape == state_dict[k].shape:
                    state_dict_suitable[k] = v
                else:
                    keys_wrong_shape.append(k)
            logging.info('Shape unmatched weights: {}'.format(keys_wrong_shape))
            msg = model.load_state_dict(state_dict_suitable, strict=False)
            logging.info(msg)
    else:
        if cfg.TEST.TEST_MODEL_PATH:
            logging.info('load test model from {}'.format(cfg.TEST.TEST_MODEL_PATH))
            msg = model.load_state_dict(torch.load(cfg.TEST.TEST_MODEL_PATH), strict=False)
            logging.info(msg)

from skimage import transform
class RescaleT(object):

        def __init__(self,output_size):
            assert isinstance(output_size,(int,tuple))
            self.output_size = output_size

        def __call__(self,sample):
            image= sample
#             print(image)
#             plt.imshow(image)
            h, w = image.shape[:2]

            if isinstance(self.output_size,int):
                if h > w:
                    new_h, new_w = self.output_size*h/w,self.output_size
                else:
                    new_h, new_w = self.output_size,self.output_size*w/h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
            # img = transform.resize(image,(new_h,new_w),mode='constant')
            # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

            img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
            # lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

            return img

class ToTensorLab(object):
        """Convert ndarrays in sample to Tensors."""
        def _init_(self,flag=0):
            self.flag = flag

        def _call_(self, sample):

            image =sample


            # change the color space
            if self.flag == 2: # with rgb and Lab colors
                tmpImg = np.zeros((image.shape[0],image.shape[1],6))
                tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
                if image.shape[2]==1:
                    tmpImgt[:,:,0] = image[:,:,0]
                    tmpImgt[:,:,1] = image[:,:,0]
                    tmpImgt[:,:,2] = image[:,:,0]
                else:
                    tmpImgt = image
                tmpImgtl = color.rgb2lab(tmpImgt)

                # nomalize image to range [0,1]
                tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
                tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
                tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
                tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
                tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
                tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

                # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
                tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
                tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
                tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

            elif self.flag == 1: #with Lab color
                tmpImg = np.zeros((image.shape[0],image.shape[1],3))

                if image.shape[2]==1:
                    tmpImg[:,:,0] = image[:,:,0]
                    tmpImg[:,:,1] = image[:,:,0]
                    tmpImg[:,:,2] = image[:,:,0]
                else:
                    tmpImg = image

                tmpImg = color.rgb2lab(tmpImg)

                # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

            else: # with rgb color
                tmpImg = np.zeros((image.shape[0],image.shape[1],3))
                image = image/np.max(image)
                if image.shape[2]==1:
                    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
                else:
                    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225


                    
            # change the r,g,b to b,r,g from [0,255] to [0,1]
            #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            tmpImg = tmpImg.transpose((2, 0, 1))

            return {'image': torch.from_numpy(tmpImg)}

from torchvision import transforms, utils

def create_tensor_from_image(url):

    image = cv2.imread(url)
    
    t1 = transforms.Compose([RescaleT(512),
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

    image_t = t1(image)
    
    inputs_test = image_t

    inputs_test = inputs_test.unsqueeze(0)  

    inputs_test = inputs_test.type(torch.FloatTensor)


    return inputs_test

# get_segmentation_backbone


import os
import torch
import logging
import torch.utils.model_zoo as model_zoo

#from ...utils.registry import Registry
#from ...config import cfg

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, i.e. resnet.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
    'hrnet_w18_small_v1': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/hrnet-w18-small-v1-08f8ae64.pth',
    'mobilenet_v2': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/mobilenetV2-15498621.pth',
}


def load_backbone_pretrained(model, backbone):
    if cfg.PHASE == 'train' and cfg.TRAIN.BACKBONE_PRETRAINED and (not cfg.TRAIN.PRETRAINED_MODEL_PATH):
        if os.path.isfile(cfg.TRAIN.BACKBONE_PRETRAINED_PATH):
            logging.info('Load backbone pretrained model from {}'.format(
                cfg.TRAIN.BACKBONE_PRETRAINED_PATH
            ))
            msg = model.load_state_dict(torch.load(cfg.TRAIN.BACKBONE_PRETRAINED_PATH), strict=False)
            logging.info(msg)
        elif backbone not in model_urls:
            logging.info('{} has no pretrained model'.format(backbone))
            return
        else:
            logging.info('load backbone pretrained model from url..')
            msg = model.load_state_dict(model_zoo.load_url(model_urls[backbone]), strict=False)
            logging.info(msg)


def get_segmentation_backbone(backbone, norm_layer=torch.nn.BatchNorm2d):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(norm_layer)
    load_backbone_pretrained(model, backbone)
    return model




# SegmentationDataset class

"""Base segmentation dataset"""
import os
import random
import numpy as np
import torchvision

from PIL import Image, ImageOps, ImageFilter
#from ...config import cfg




class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()
        self.root = os.path.join(cfg.ROOT_PATH, root)
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = self.to_tuple(crop_size)
        self.color_jitter = self._get_color_jitter()

    def to_tuple(self, size):
        if isinstance(size, (list, tuple)):
            return tuple(size)
        elif isinstance(size, (int, float)):
            return tuple((size, size))
        else:
            raise ValueError('Unsupport datatype: {}'.format(type(size)))

    def _get_color_jitter(self):
        color_jitter = cfg.AUG.COLOR_JITTER
        if color_jitter is None:
            return None
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        return torchvision.transforms.ColorJitter(*color_jitter)

    def _val_sync_transform(self, img, mask):
        short_size = self.base_size
        img = img.resize((short_size, short_size), Image.BILINEAR)
        mask = mask.resize((short_size, short_size), Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        short_size = self.base_size
        img = img.resize((short_size, short_size), Image.BILINEAR)
        mask = mask.resize((short_size, short_size), Image.NEAREST)
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


   
 # TransExtraSegmentation class

"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
#from .seg_data_base import SegmentationDataset
from IPython import embed

class TransExtraSegmentation(SegmentationDataset):
    """Trans10K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trans10K folder. Default is './datasets/Trans10K'
    split: string
        'train', 'validation', 'test'
    transform : callable, optional
        A function that transforms the image
    """
    BASE_DIR = 'Trans10K'
    NUM_CLASS = 3

    def __init__(self, root='demo/imgs', split='train', mode=None, transform=None, **kwargs):
        super(TransExtraSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        '''print(f"self.root: {self.root}")
        assert os.path.exists(self.root), "Please put dataset in {SEG_ROOT}/datasets/Extra"
        self.images = _get_demo_pairs(self.root)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")'''

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = np.zeros_like(np.array(img))[:,:,0]
        assert mask.max()<=2, mask.max()
        mask = Image.fromarray(mask)

        # synchrosized transform
        img, mask = self._val_sync_transform(img, mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, self.images[index]

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'things', 'stuff')


def _get_demo_pairs(folder):

    def get_path_pairs(img_folder):
        img_paths = []
        imgs = os.listdir(img_folder)
        for imgname in imgs:
            imgpath = os.path.join(img_folder, imgname)
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
            else:
                logging.info('cannot find the image:', imgpath)

        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths

    img_folder = folder
    img_paths = get_path_pairs(img_folder)

    return img_paths

    
datasets = {
    'trans10k_extra': TransExtraSegmentation()
}





# this code heavily based on detectron2
import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            # logger = logging.getLogger(__name__)
            logging.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data + module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def groupNorm(num_channels, eps=1e-5, momentum=0.1, affine=True):
    return nn.GroupNorm(min(32, num_channels), num_channels, eps=eps, affine=affine)


def get_norm(norm):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    support_norm_type = ['BN', 'SyncBN', 'FrozenBN', 'GN', 'nnSyncBN']
    assert norm in support_norm_type, 'Unknown norm type {}, support norm types are {}'.format(
                                                                        norm, support_norm_type)
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": groupNorm,
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

# Segbasemodel

"""Base Model for Semantic Segmentation"""
import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from .backbones import get_segmentation_backbone
#from ..data.dataloader import datasets
#from ..modules import get_norm
#from ..config import cfg


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    """
    def __init__(self, need_backbone=True):
        super(SegBaseModel, self).__init__()
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = cfg.SOLVER.AUX
        self.norm_layer = get_norm(cfg.MODEL.BN_TYPE)
        self.backbone = None
        self.encoder = None
        if need_backbone:
            self.get_backbone()

    def get_backbone(self):
        self.backbone = cfg.MODEL.BACKBONE.lower()
        self.encoder = get_segmentation_backbone(self.backbone, self.norm_layer)

    def base_forward(self, x):
        """forwarding backbone network"""
        c1, c2, c3, c4 = self.encoder(x)
        return c1, c2, c3, c4

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

    def evaluate(self, image):
        """evaluating network with inputs and targets"""
        scales = cfg.TEST.SCALES
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = torch.zeros((batch, self.nclass, h, w)).to(image.device)
        scores = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            outputs = self.forward(cur_img)[0][..., :height, :width]

            score = _resize_image(outputs, h, w)

            if scores is None:
                scores = score
            else:
                scores += score
        return scores


def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)


def _pad_image(img, crop_size):
    b, c, h, w = img.shape
    assert(c == 3)
    padh = crop_size[0] - h if h < crop_size[0] else 0
    padw = crop_size[1] - w if w < crop_size[1] else 0
    if padh == 0 and padw == 0:
        return img
    img_pad = F.pad(img, (0, padh, 0, padw))

    # TODO clean this code
    # mean = cfg.DATASET.MEAN
    # std = cfg.DATASET.STD
    # pad_values = -np.array(mean) / np.array(std)
    # img_pad = torch.zeros((b, c, h + padh, w + padw)).to(img.device)
    # for i in range(c):
    #     # print(img[:, i, :, :].unsqueeze(1).shape)
    #     img_pad[:, i, :, :] = torch.squeeze(
    #         F.pad(img[:, i, :, :].unsqueeze(1), (0, padh, 0, padw),
    #               'constant', value=pad_values[i]), 1)
    # assert(img_pad.shape[2] >= crop_size[0] and img_pad.shape[3] >= crop_size[1])

    return img_pad


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert(img.ndim == 4)
    return img.flip((3))


def _to_tuple(size):
    if isinstance(size, (list, tuple)):
        assert len(size), 'Expect eval crop size contains two element, '                           'but received {}'.format(len(size))
        return tuple(size)
    elif isinstance(size, numbers.Number):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))



"""Basic Module for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from IPython import embed

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True,
                 bias=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size,
                              stride=stride, padding=dilation,
                              dilation=dilation, groups=inplanes, bias=bias)
        bn_depth = norm_layer(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)
        bn_point = norm_layer(planes)

        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()),
                                                    ('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point)
                                                    ]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise),
                                                    ('bn_depth', bn_depth),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('pointwise', pointwise),
                                                    ('bn_point', bn_point),
                                                    ('relu2', nn.ReLU(inplace=True))
                                                    ]))

    def forward(self, x):
        return self.block(x)
    
    
"""Basic Module for Semantic Segmentation"""




class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------
#                      For deeplab
# -----------------------------------------------------------------
class _ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        elif output_stride == 32:
            dilations = [6, 12, 18]
        else:
            raise NotImplementedError

        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                ('bn', nn.BatchNorm2d(out_channels)),
                                                ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)

        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))),
                                                        ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)),
                                                        ('bn', nn.BatchNorm2d(out_channels)),
                                                        ('relu', nn.ReLU(inplace=True))]))

        self.conv = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)

        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

import torch.nn as nn

#from .build import BACKBONE_REGISTRY
#from ...config import cfg

__all__ = ['ResNetV1']


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):
        output_stride = cfg.MODEL.OUTPUT_STRIDE
        scale = cfg.MODEL.BACKBONE_SCALE
        if output_stride == 32:
            dilations = [1, 1]
            strides = [2, 2]
        elif output_stride == 16:
            dilations = [1, 2]
            strides = [2, 1]
        elif output_stride == 8:
            dilations = [2, 4]
            strides = [1, 1]
        else:
            raise NotImplementedError
        self.inplanes = int((128 if deep_stem else 64) * scale)
        super(ResNetV1, self).__init__()
        if deep_stem:
            # resnet vc
            mid_channel = int(64 * scale)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, mid_channel, 3, 2, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
                norm_layer(mid_channel),
                nn.ReLU(True),
                nn.Conv2d(mid_channel, self.inplanes, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, int(64 * scale), layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, int(128 * scale), layers[1], stride=2, norm_layer=norm_layer)

        self.layer3 = self._make_layer(block, int(256 * scale), layers[2], stride=strides[0], dilation=dilations[0],
                                       norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, int(512 * scale), layers[3], stride=strides[1], dilation=dilations[1],
                                       norm_layer=norm_layer, multi_grid=cfg.MODEL.DANET.MULTI_GRID,
                                       multi_dilation=cfg.MODEL.DANET.MULTI_DILATION)

        self.last_inp_channels = int(512 * block.expansion * scale)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * block.expansion * scale), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckV1b):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1b):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,
                    multi_grid=False, multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if not multi_grid:
            if dilation in (1, 2):
                layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion

        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div],
                                    previous_dilation=dilation, norm_layer=norm_layer))
        else:
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation,
                                    previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # for classification
        # x = self.avgpool(c4)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return c1, c2, c3, c4


@BACKBONE_REGISTRY.register()
def resnet18(norm_layer=nn.BatchNorm2d):
    num_block = [2, 2, 2, 2]
    return ResNetV1(BasicBlockV1b, num_block, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet34(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 6, 3]
    return ResNetV1(BasicBlockV1b, num_block, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet50(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 6, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet101(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 23, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet152(norm_layer=nn.BatchNorm2d):
    num_block = [3, 8, 36, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer)


@BACKBONE_REGISTRY.register()
def resnet50c(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 6, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)


@BACKBONE_REGISTRY.register()
def resnet101c(norm_layer=nn.BatchNorm2d):
    num_block = [3, 4, 23, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)


@BACKBONE_REGISTRY.register()
def resnet152c(norm_layer=nn.BatchNorm2d):
    num_block = [3, 8, 36, 3]
    return ResNetV1(BottleneckV1b, num_block, norm_layer=norm_layer, deep_stem=True)



import torch
import torch.nn as nn
import torch.nn.functional as F

#from .segbase import SegBaseModel
#from .model_zoo import MODEL_REGISTRY
#from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
#from ..config import cfg
from IPython import embed
import math

__all__ = ['TransLab']

def _resize_image(img, h, w):
    return F.interpolate(img, size=[h, w], mode='bilinear', align_corners=True)

@MODEL_REGISTRY.register(name='TransLab')
class TransLab(SegBaseModel):
    def __init__(self):
        super(TransLab, self).__init__()
        if self.backbone.startswith('mobilenet'):
            c1_channels = 24
            c4_channels = 320
        else:
            c1_channels = 256
            c4_channels = 2048
            c2_channel = 512

        self.head = _DeepLabHead_attention(self.nclass, c1_channels=c1_channels, c4_channels=c4_channels, c2_channel=c2_channel)
        self.head_b = _DeepLabHead(1, c1_channels=c1_channels, c4_channels=c4_channels)

        self.fus_head1 = FusHead()
        self.fus_head2 = FusHead(inplane=2048)
        self.fus_head3 = FusHead(inplane=512)

        if self.aux:
            self.auxlayer = _FCNHead(728, self.nclass)
        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        c1, c2, c3, c4 = self.encoder(x)
        outputs = list()
        outputs_b = list()

        x_b = self.head_b(c4, c1)

        #attention c1 c4
        attention_map = x_b.sigmoid()

        c1 = self.fus_head1(c1, attention_map)
        c4 = self.fus_head2(c4, attention_map)
        c2 = self.fus_head3(c2, attention_map)

        x = self.head(c4, c2, c1, attention_map)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x_b = F.interpolate(x_b, size, mode='bilinear', align_corners=True)

        outputs.append(x)
        outputs_b.append(x_b)#.sigmoid())

        return tuple(outputs), tuple(outputs_b)

    def evaluate(self, image):
        """evaluating network with inputs and targets"""
        scales = cfg.TEST.SCALES
        batch, _, h, w = image.shape
        base_size = max(h, w)
        # scores = torch.zeros((batch, self.nclass, h, w)).to(image.device)
        scores = None
        scores_boundary = None
        for scale in scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)

            # resize image to current size
            cur_img = _resize_image(image, height, width)
            outputs, outputs_boundary = self.forward(cur_img)
            outputs = outputs[0][..., :height, :width]
            outputs_boundary = outputs_boundary[0][..., :height, :width]

            score = _resize_image(outputs, h, w)
            score_boundary = _resize_image(outputs_boundary, h, w)

            if scores is None:
                scores = score
                scores_boundary = score_boundary
            else:
                scores += score
                scores_boundary += score_boundary
        return scores, scores_boundary


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        # self.use_aspp = cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP
        # self.use_decoder = cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER
        self.use_aspp = True
        self.use_decoder = True
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48
        self.block = nn.Sequential(
            SeparableConv2d(last_channels, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)
        if self.use_decoder:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            cat_fmap = torch.cat([x, c1], dim=1)
            return self.block(cat_fmap)

        return self.block(x)


class _DeepLabHead_attention(nn.Module):
    def __init__(self, nclass, c1_channels=256, c4_channels=2048, c2_channel=512, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead_attention, self).__init__()
        # self.use_aspp = cfg.MODEL.DEEPLABV3_PLUS.USE_ASPP
        # self.use_decoder = cfg.MODEL.DEEPLABV3_PLUS.ENABLE_DECODER
        self.use_aspp = True
        self.use_decoder = True
        last_channels = c4_channels
        if self.use_aspp:
            self.aspp = _ASPP(c4_channels, 256)
            last_channels = 256
        if self.use_decoder:
            self.c1_block = _ConvBNReLU(c1_channels, 48, 1, norm_layer=norm_layer)
            last_channels += 48

            self.c2_block = _ConvBNReLU(c2_channel, 24, 1, norm_layer=norm_layer)
            last_channels += 24

        self.block = nn.Sequential(
            SeparableConv2d(256+24+48, 256, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256, 256, 3, norm_layer=norm_layer, relu_first=False),
            nn.Conv2d(256, nclass, 1))

        self.block_c2 = nn.Sequential(
            SeparableConv2d(256+24, 256+24, 3, norm_layer=norm_layer, relu_first=False),
            SeparableConv2d(256+24, 256+24, 3, norm_layer=norm_layer, relu_first=False))


        self.fus_head_c2 = FusHead(inplane=256+24)
        self.fus_head_c1 = FusHead(inplane=256+24+48)


    def forward(self, x, c2, c1, attention_map):
        c1_size = c1.size()[2:]
        c2_size = c2.size()[2:]
        if self.use_aspp:
            x = self.aspp(x)


        if self.use_decoder:
            x = F.interpolate(x, c2_size, mode='bilinear', align_corners=True)
            c2 = self.c2_block(c2)
            x = torch.cat([x, c2], dim=1)
            x = self.fus_head_c2(x, attention_map)
            x = self.block_c2(x)

            x = F.interpolate(x, c1_size, mode='bilinear', align_corners=True)
            c1 = self.c1_block(c1)
            x = torch.cat([x, c1], dim=1)
            x = self.fus_head_c1(x, attention_map)
            return self.block(x)

        return self.block(x)


class FusHead(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, inplane=256):
        super(FusHead, self).__init__()
        self.conv1 = SeparableConv2d(inplane*2, inplane, 3, norm_layer=norm_layer, relu_first=False)
        self.fc1 = nn.Conv2d(inplane, inplane // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(inplane // 16, inplane, kernel_size=1)

    def forward(self, c, att_map):
        if c.size() != att_map.size():
            att_map  = F.interpolate(att_map, c.size()[2:], mode='bilinear', align_corners=True)

        atted_c = c * att_map
        x = torch.cat([c, atted_c], 1)#512
        x = self.conv1(x) #256

        weight = F.avg_pool2d(x, x.size(2))
        weight = F.relu(self.fc1(weight))
        weight = torch.sigmoid(self.fc2(weight))
        x = x * weight
        return x















#from __future__ import print_function

import os
import sys

#cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = ""
#sys.path.append(root_path)

# print(f"cur_path: {cur_path}")
# print(f"root_path: {root_path}")

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms

#from segmentron.models.model_zoo import get_segmentation_model
#from segmentron.config import cfg
#from segmentron.utils.options import parse_args
#from segmentron.utils.default_setup import default_setup
from IPython import embed
from collections import OrderedDict

import cv2
import numpy as np

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')

        # image transform
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
                logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
                self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)
        self.count_easy = 0
        self.count_hard = 0
    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)
    import time
    def eval(self, url):   
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        import matplotlib.pyplot as plt
        
        image = create_tensor_from_image(url)
        image = image.to(self.device)
        with torch.no_grad():
            
            output, output_boundary = model.evaluate(image)
            ori_img = cv2.imread(url)
            h, w, _ = ori_img.shape

            glass_res = output.argmax(1)[0].data.cpu().numpy().astype('uint8') * 127
            glass_res = cv2.resize(glass_res, (w, h), interpolation=cv2.INTER_NEAREST)
            
            return glass_res
        


class RescaleT(object):

        def __init__(self,output_size):
            assert isinstance(output_size,(int,tuple))
            self.output_size = output_size

        def __call__(self,sample):
            image= sample

            h, w = image.shape[:2]

            if isinstance(self.output_size,int):
                if h > w:
                    new_h, new_w = self.output_size*h/w,self.output_size
                else:
                    new_h, new_w = self.output_size,self.output_size*w/h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
            # img = transform.resize(image,(new_h,new_w),mode='constant')
            # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

            img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
            # lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

            return img

class ToTensorLab(object):
        """Convert ndarrays in sample to Tensors."""
        def __init__(self,flag=0):
            self.flag = flag

        def __call__(self, sample):

            image =sample


            # change the color space
            if self.flag == 2: # with rgb and Lab colors
                tmpImg = np.zeros((image.shape[0],image.shape[1],6))
                tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
                if image.shape[2]==1:
                    tmpImgt[:,:,0] = image[:,:,0]
                    tmpImgt[:,:,1] = image[:,:,0]
                    tmpImgt[:,:,2] = image[:,:,0]
                else:
                    tmpImgt = image
                tmpImgtl = color.rgb2lab(tmpImgt)

                # nomalize image to range [0,1]
                tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
                tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
                tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
                tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
                tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
                tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

                # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
                tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
                tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
                tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

            elif self.flag == 1: #with Lab color
                tmpImg = np.zeros((image.shape[0],image.shape[1],3))

                if image.shape[2]==1:
                    tmpImg[:,:,0] = image[:,:,0]
                    tmpImg[:,:,1] = image[:,:,0]
                    tmpImg[:,:,2] = image[:,:,0]
                else:
                    tmpImg = image

                tmpImg = color.rgb2lab(tmpImg)

                # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

                tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
                tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
                tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

            else: # with rgb color
                tmpImg = np.zeros((image.shape[0],image.shape[1],3))
                image = image/np.max(image)
                if image.shape[2]==1:
                    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
                else:
                    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225


            # change the r,g,b to b,r,g from [0,255] to [0,1]
            #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            tmpImg = tmpImg.transpose((2, 0, 1))

            return {'image': torch.from_numpy(tmpImg)}




def get_mask(image_url, evaluator):

    image = io.imread(image_url)

    # Remove below line after testing
    #image_temp = image
    #plt.axis('off')
    #plt.imshow(image)

    #import time
    #ts = time.time()
    image_map = evaluator.eval(image_url)
    #te = time.time()
    image_map = cv2.cvtColor(image_map, cv2.COLOR_BGR2RGB)
    
    return image_map



#if __name__ == '__main__':
   
    #args = initiate_process()
    #evaluator = Evaluator(args)

