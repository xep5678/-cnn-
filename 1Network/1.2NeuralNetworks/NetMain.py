# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:29:17 2022

@author: admin
"""
import random
from importlib import import_module
import torch
import numpy as np
from option import args
import common
from trainer import Trainer


def main(data_loader_path, length_freq, num_of_receiver, receiver_depth, cpu, rest_time, seed, num_of_sources, model_name,
         batch_size, batch_size_v, batch_size_exp, mini_epoch, max_epoch, test_only, plot_only, exp_only, resume,
         load_model_path, lr, save_file):
    args.data_loader_path = data_loader_path
    args.length_freq = length_freq
    args.num_of_receiver = num_of_receiver
    args.receiver_depth = receiver_depth
    args.cpu = cpu
    args.rest_time = rest_time
    args.seed = seed
    args.num_of_sources = num_of_sources
    args.model = model_name
    args.batch_size = batch_size
    args.batch_size_v = batch_size_v
    args.batch_size_exp = batch_size_exp
    args.mini_epoch = mini_epoch
    args.max_epoch = max_epoch
    args.test_only = test_only
    args.plot_only = plot_only
    args.exp_only = exp_only
    args.resume = resume
    args.load_model_path = load_model_path
    args.lr = lr
    args.save_file = save_file

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    #Checkpoint是用于描述在每次训练后保存模型参数（权重）的惯例或术语。
    checkpoint = common.CheckPoint(args)
    module = import_module('model.' + args.model.lower())
    current_model = module.make_model(args.length_freq)
    t = Trainer(args, current_model, checkpoint)
    if (not args.test_only) and (not args.exp_only) and (not args.plot_only):
        model_path = t.train(args.load_model_path)
        t.test(model_path)
        t.exp_deal(model_path)
        model_save_name = (str.split(model_path, '/')[-1])[:-4]
    elif args.test_only:
        t.test(args.load_model_path)
        model_save_name = (str.split(args.load_model_path, '/')[-1])[:-4]
    elif args.exp_only:
        t.exp_deal(args.load_model_path)
        model_save_name = (str.split(args.load_model_path, '/')[-1])[:-4]
    elif args.plot_only:
        t.plotloss(args.load_model_path)
        model_save_name = (str.split(args.load_model_path, '/')[-1])[:-4]

    checkpoint.done()

    return model_save_name
