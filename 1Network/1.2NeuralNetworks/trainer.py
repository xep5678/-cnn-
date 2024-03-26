# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:48:02 2022

@author: admin
"""
import os
# import math
# from decimal import Decimal
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
import time
import pylab as pl
from Plt.beaplot2 import plot as bp
from data.Dataloader import SnSpectrumLoader
import scipy.io as sio
import glob

class Trainer():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.ckp = ckp
        self.model = my_model
        self.num_workers = args.num_workers
        self.rest_time = args.rest_time  # Machine rest time for one epoch (s)

    def train(self, save_name_weight_para_file, mode='train'):
        # Use gpu to accelerate the computation process
        # torch.cuda.empty_cache()可以释放内存空间
        torch.cuda.empty_cache()
        # 构建的张量或者模型被分配到对应的设备上
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("running at device %s" % default_device)
        default_type = torch.float32
        self.model = self.model.to(default_device).type(default_type)

        # Training set, test set, experimental data, model file path
        model_name = self.args.model

        # path = os.path.abspath('../1.1 Dataset Simulation/1.1.3 Dataset/1.1.3.1 Training Set/' + self.args.data_train)
        path = os.path.abspath(self.args.data_loader_path + '/(3)Dataset/A.TrainingSet')
        path_v = os.path.abspath(self.args.data_loader_path + '/(3)Dataset/B.ValidationSet')
        # glob.glob返回的是所有匹配的文件路径列表（list）
        file_name_all = glob.glob(path + '/*.sim')
        appendix = str.split(self.args.data_loader_path, '/')[-1]

        data_size = os.path.getsize(file_name_all[0]) * len(file_name_all)
        num_of_frequency = self.args.length_freq
        num_of_receiver = self.args.num_of_receiver
        #sim文件中，一个距离、深度对应的数据量，最后的“+2”应当指的是 距离和深度
        one_sample_data_size = 2 * num_of_frequency * num_of_receiver + 2  # One sample data size

        batch_size = self.args.batch_size  # Training batch size
        batch_size_val = self.args.batch_size_v  # Validation batch size

        # Model initialization
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        print("Total number of parameters in networks is {} ".format(
            sum(x.numel() for x in self.model.parameters())))  # Print number of network parameters

        # Normalized value
        max_range = 20
        max_depth = 100

        # Set Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=5e-5)

        # Load model or reset model
        # resume代表的是是否要重复训练
        if self.args.resume:
            path_checkpoint = self.args.load_model_path # Breakpoint paths
            checkpoint = torch.load(path_checkpoint)  # Load breakpoints
            self.model.load_state_dict(checkpoint['model'])  # Load model learnable parameters
            optimizer.load_state_dict(checkpoint['optimizer'])  # Load Optimizer Parameters
            mini_epoch = self.args.mini_epoch
            max_epoch = self.args.max_epoch
            start_epoch = checkpoint['current_epoch'] + 1

            # Loss and MAE monitored by epoch
            training_loss = checkpoint['training_loss']
            validation_loss = checkpoint['validation_loss']
            # Variables in training process
            log_sigma_of_range = checkpoint['log_sigma_of_range']
            log_sigma_of_depth = checkpoint['log_sigma_of_depth']
            MAPE_of_range = checkpoint['MAPE_of_range']
            MAPE_of_depth = checkpoint['MAPE_of_depth']
            MAE_of_range = checkpoint['MAE_of_range']
            MAE_of_depth = checkpoint['MAE_of_depth']
            # Variables in validation process
            MAPE_of_range_v = checkpoint['MAPE_of_range_v']
            MAPE_of_depth_v = checkpoint['MAPE_of_depth_v']
            MAE_of_range_v = checkpoint['MAE_of_range_v']
            MAE_of_depth_v = checkpoint['MAE_of_depth_v']

        else:
            start_epoch = 0
            save_name_weight_para_file = ''  # Initialization model path
            mini_epoch = self.args.mini_epoch
            max_epoch = self.args.max_epoch
            # Loss and MAE monitored by epoch
            training_loss = []
            validation_loss = []
            # Variables in training process
            log_sigma_of_range = []
            log_sigma_of_depth = []
            MAPE_of_range = []
            MAPE_of_depth = []
            MAE_of_range = []
            MAE_of_depth = []
            # Variables in validation process
            MAPE_of_range_v = []
            MAPE_of_depth_v = []
            MAE_of_range_v = []
            # stop error
            MAE_of_depth_v = []

        # 一个float32占4字节
        batch_ndx = int(data_size / one_sample_data_size / 4 / batch_size / mini_epoch)
        num_of_sources = self.args.num_of_sources
        num_of_sources_v = int(num_of_sources * batch_size_val / batch_size) + 1
        Nba = int(batch_ndx * batch_size / num_of_sources)

        # Start training
        for current_epoch in range(start_epoch, max_epoch * mini_epoch):

            # Set the training process signal-to-noise ratio
            if current_epoch <= 1:
                # In the first epoch, we want the SNR to be a little small.
                SNR_range_of_train = [-10, 10]
                SNR_range_of_validation = [-20, 10]
            else:
                SNR_range_of_train = [20, 30]
                SNR_range_of_validation = [10, 20]

            # Initialization parameters
            loss = 0
            MAPE_r = 0  # MAPE of range estimation in training process
            MAPE_d = 0  # MAPE of depth estimation
            MAE_r = 0  # MAE of range estimation (km)
            MAE_d = 0  # MAE of depth estimation (m)
            loss_v = 0
            MAPE_r_v = 0  # MAPE of range estimation in validation process
            MAPE_d_v = 0  # MAPE of depth estimation in validation process
            MAE_r_v = 0  # MAE of range estimation in validation process (km)
            MAE_d_v = 0  # MAE of detph estimation in validation process (m)

            for iba in range(0, Nba):

                # Data loading
                #设置训练集
                train_set = SnSpectrumLoader(file_path=path, length_freq=num_of_frequency,
                                             SNR_range=SNR_range_of_train,
                                             run_mode=mode, model=model_name, num_of_receiver=num_of_receiver)
                #对训练集train_set的加载方式
                train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=self.num_workers,
                                               drop_last=True)
                val_set = SnSpectrumLoader(file_path=path_v, length_freq=num_of_frequency,
                                           SNR_range=SNR_range_of_validation,
                                           run_mode=mode, model=model_name, num_of_receiver=num_of_receiver)
                val_loader = data.DataLoader(val_set, batch_size=batch_size_val, shuffle=True,
                                             num_workers=self.num_workers,
                                             drop_last=True)
                for batch_idx_mini, dataTrain in enumerate(train_loader):

                    batch_idx = batch_idx_mini + iba * len(train_loader)
                    self.model.train()

                    # Import data
                    inputs = dataTrain['C'].float()
                    inputs = Variable(inputs)
                    inputs = inputs.to(default_device).type(default_type)
                    r = dataTrain['r'].float() / max_range  # Range target
                    r = r.to(default_device).type(default_type)
                    d = dataTrain['z'].float() / max_depth  # Depth target
                    d = d.to(default_device).type(default_type)
                    optimizer.zero_grad()

                    # Input data into the network for training
                    try:
                        train_loss, log_vars, output = self.model(inputs, [r, d])
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('|WARNING: run out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e

                    train_loss.backward()
                    optimizer.step()

                    # Calculation error
                    MAPE_r += float(((output[0] - r).abs() / r).sum(0) / batch_size)
                    MAPE_d += float(((output[1] - d).abs() / d).sum(0) / batch_size)
                    MAE_r += float((output[0] - r).abs().sum(0) / batch_size * max_range)
                    MAE_d += float((output[1] - d).abs().sum(0) / batch_size * max_depth)
                    loss += float(train_loss.item())

                    # Validation process
                    with torch.no_grad():
                        self.model.eval()
                        _, dataVali = list(enumerate(val_loader))[batch_idx_mini]
                        inputs_v = dataVali['C'].float()
                        inputs_v = Variable(inputs_v)
                        inputs_v = inputs_v.to(default_device).type(default_type)
                        r_v = dataVali['r'].float() / max_range
                        r_v = r_v.to(default_device).type(default_type)
                        z_v = dataVali['z'].float() / max_depth
                        z_v = z_v.to(default_device).type(default_type)

                        val_loss, log_vars_v, output_v = self.model(inputs_v, [r_v, z_v])

                        MAPE_r_v += float(((output_v[0] - r_v).abs() / r_v).sum(0) / batch_size_val)
                        MAPE_d_v += float(((output_v[1] - z_v).abs() / z_v).sum(0) / batch_size_val)
                        MAE_r_v += float((output_v[0] - r_v).abs().sum(0) / batch_size_val * max_range)
                        MAE_d_v += float((output_v[1] - z_v).abs().sum(0) / batch_size_val * max_depth)
                        loss_v += float(val_loss.item())

                    # print
                    if batch_idx % 10 == 9:
                        # print(batch_idx + 1, batch_ndx,
                        #       'Train Loss: %.3f |MAE_R: %.3fkm MAE_Z: %.3fm'
                        #       % (loss / (batch_idx + 1),
                        #          MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1)))
                        # print('Validation Loss: %.3f |MAE_R: %.3fkm MAE_Z: %.3fm\n'
                        #       % (val_loss / (batch_idx + 1),
                        #          MAE_r_v / (batch_idx + 1), MAE_d_v / (batch_idx + 1)))
                        print('{:d} {:d} Train Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m'.
                                   format(batch_idx + 1, batch_ndx, loss / (batch_idx + 1),
                                          MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1)))

                        print('Validation Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m\n'.
                                   format(loss_v / (batch_idx + 1),
                                          MAE_r_v / (batch_idx + 1), MAE_d_v / (batch_idx + 1)))

                # Release variables
                del dataTrain
                del train_set
                del train_loader
                del val_set
                del val_loader
                del inputs
                del r
                del d
                del train_loss
                del output
                del dataVali
                del inputs_v
                del r_v
                del z_v
                del val_loss
                del output_v

            self.ckp.write_log(
                ('{:d} {:d} Train Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m'.
                 format(batch_idx + 1, batch_ndx, loss / (batch_idx + 1),
                        MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1))))
            self.ckp.write_log(
                ('Validation Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m\n'.
                 format(loss_v / (batch_idx + 1),
                        MAE_r_v / (batch_idx + 1), MAE_d_v / (batch_idx + 1))))

            # Save variables to the list
            log_sigma_of_range.append(log_vars[0])
            log_sigma_of_depth.append(log_vars[1])
            MAPE_of_range.append(100 * MAPE_r / (batch_idx + 1))
            MAPE_of_depth.append(100 * MAPE_d / (batch_idx + 1))
            MAE_of_range.append(MAE_r / (batch_idx + 1))
            MAE_of_depth.append(MAE_d / (batch_idx + 1))
            training_loss.append(loss / (batch_idx + 1))

            MAPE_of_range_v.append(100 * MAPE_r_v / (batch_idx + 1))
            MAPE_of_depth_v.append(100 * MAPE_d_v / (batch_idx + 1))
            MAE_of_range_v.append(MAE_r_v / (batch_idx + 1))
            MAE_of_depth_v.append(MAE_d_v / (batch_idx + 1))
            validation_loss.append(loss_v / (batch_idx + 1))

            save_info = {  # Saved information
                "current_epoch": current_epoch,  # Number of iterative steps
                "optimizer": optimizer.state_dict(),
                "model": self.model.state_dict(),
                'mini_epoch': mini_epoch,
                'max_epoch': max_epoch,
                'training_loss': training_loss,
                'log_sigma_of_range': log_sigma_of_range,
                'log_sigma_of_depth': log_sigma_of_depth,
                'MAPE_of_range': MAPE_of_range,
                'MAPE_of_depth': MAPE_of_depth,
                'MAE_of_range': MAE_of_range,
                'MAE_of_depth': MAE_of_depth,
                'validation_loss': validation_loss,
                'MAPE_of_range_v': MAPE_of_range_v,
                'MAPE_of_depth_v': MAPE_of_depth_v,
                'MAE_of_range_v': MAE_of_range_v,
                'MAE_of_depth_v': MAE_of_depth_v,
            }
            filepath_weight_para_save_folder = os.path.abspath(self.args.save_file + '/a.weight_parameter')
            if not os.path.exists(filepath_weight_para_save_folder):
                os.makedirs(filepath_weight_para_save_folder)
            if save_name_weight_para_file != '' and ('.00' not in save_name_weight_para_file):
                # Delete the last weight file
                os.remove(filepath_weight_para_save_folder + '/' + save_name_weight_para_file + '.pth')

            save_name_weight_para_file = ('{}_{}_epoch_{:.2f}'
                                          .format(model_name, appendix,
                                                  (current_epoch + 1) / mini_epoch))
            save_path_weight_para_file = filepath_weight_para_save_folder + '/' + save_name_weight_para_file + '.pth'
            torch.save(save_info, save_path_weight_para_file)

            # Release variables
            del log_vars
            del MAPE_d
            del MAPE_r
            del MAE_d
            del MAE_r
            del loss
            del log_vars_v
            del MAPE_d_v
            del MAPE_r_v
            del MAE_d_v
            del MAE_r_v
            del loss_v

            # Rest and wait for the machine temperature to drop
            time.sleep(self.rest_time)

        # Plot and save
        filepath_figure_save_folder = os.path.abspath(self.args.save_file + '/b.training_results/figure')
        save_name_figure_file = save_name_weight_para_file + '_training_process'
        save_path_figure_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.svg'

        # save_file = './results/training_results/images'
        # save_dict = save_file + '/' + model_path + '.svg'
        if not os.path.exists(filepath_figure_save_folder):
            
            os.makedirs(filepath_figure_save_folder)
        sigma_range = 10 ** np.array(log_sigma_of_range)
        sigma_depth = 10 ** np.array(log_sigma_of_depth)

        dpi = 200
        x = torch.arange(0, max_epoch * mini_epoch, 1) / mini_epoch
        pl.figure(dpi=dpi, figsize=(10, 3))
        pl.subplot(131)
        bp(x, MAE_of_range, x_tick_off=False, legend_label='Train', color='b')
        bp(x, MAE_of_range_v, title='', x_label='epoch',
           y_label=r'$MAE_r$ (km)', x_tick_off=False, legend_label='Validation', color='r',
           font_size=20, loc='upper right')
        pl.subplot(132)
        bp(x, MAE_of_depth, x_tick_off=False, legend_label='Train', color='b')
        bp(x, MAE_of_depth_v, title='', x_label='epoch',
           y_label=r'$MAE_d$ (m)', x_tick_off=False, legend_label='Validation', color='r'
           , font_size=20, loc='upper right')
        pl.subplot(133)
        bp(x, sigma_range, legend_label=r'$\sigma_r$')
        bp(x, sigma_depth, line_style='--', color='r', title='', x_label='epoch',
           y_label=r'$\sigma$', x_tick_off=False, legend_label=r'$\sigma_z$', font_size=20, loc='upper right')
        pl.tight_layout()
        #将每个周期的误差情况等的作图保留下来
        pl.savefig(save_path_figure_file, dpi=dpi, bbox_inches='tight')
        # pl.show()

        # save mat data
        filepath_data_save_folder = os.path.abspath(self.args.save_file + '/b.training_results/data')
        save_name_data_file = save_name_figure_file
        save_path_data_file = filepath_data_save_folder + '/' + save_name_data_file + '.mat'
        # save_file = './results/training_results/data'
        if not os.path.exists(filepath_data_save_folder):
            os.makedirs(filepath_data_save_folder)
        # save_dict = save_file + '/' + save_name_weight_para_file + '.mat'
        sio.savemat(save_path_data_file,
                    {'model_name': model_name,
                     'epoch': x.numpy(), 'MAE_r': np.array(MAE_of_range),
                     'MAE_d': np.array(MAE_of_depth),
                     'MAE_r_val': np.array(MAE_of_range_v),
                     'MAE_d_val': np.array(MAE_of_depth_v),
                     'sigma_range': np.array(sigma_range),
                     'sigma_depth': np.array(sigma_depth),
                     })
        print('------- Finished Training! -------')
        return save_path_weight_para_file

    def test(self, model_path, mode='test'):
        import time

        model_name = self.args.model
        self.ckp.write_log('--------Begin Evaluation-------\n')
        # Using gpu to accelerate devices
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("running at device %s" % default_device)
        default_type = torch.float32

        length_freq = self.args.length_freq
        num_of_receiver = self.args.num_of_receiver
        # path = os.path.abspath('../1.1 Dataset Simulation/1.1.3 Dataset/1.1.3.2 Test Set/' + self.args.data_test)
        path = os.path.abspath(self.args.data_loader_path + '/(3)Dataset/C.TestSet')

        # Normalized
        r_max = 20
        z_max = 100

        # Importing Models
        self.model.to(default_device).type(default_type)
        self.model.eval()
        print(
            "Total number of parameters in networks is {} ".format(sum(x.numel() for x in self.model.parameters())))

        # Loading Network
        # path_checkpoint = './weight_parameter/' + model_path + '.pth'
        path_checkpoint = model_path
        checkpoint = torch.load(path_checkpoint)
        self.model.load_state_dict(checkpoint['model'])  # -*- coding: utf-8 -*-

        # ################ Sensitivity analysis ####################
        #np.arange(起点值，终点值，步长)
        Sr = np.arange(1, 20.1, 1)
        Sd = np.arange(10, 60, 10)
        # Sd = np.array([10, 50])  # Source range and depth
        SrExpand = np.expand_dims(Sr, axis=0).repeat(len(Sd), axis=0).reshape(-1)
        SdExpand = np.expand_dims(Sd, axis=1).repeat(len(Sr), axis=1).reshape(-1)
        # print('SrExpand',SrExpand)
        # print('SdExpand', SdExpand)
        NDA = len(Sr) * len(Sd)

        ErrorLimit_Range = 0.10  # Range estimation relative error Limit
        # ErrorLimit_Range = 2  # Range estimation relative error Limit
        ErrorLimit_Depth = 5  # Depth estimation absolute error limit

        SNR = 10  # Signal-to-noise ratio
        # sen_para = np.arange(-20, 20, 1)
        file_name_all = glob.glob(path + '/*.sim')
        sen_para = np.arange(0, len(file_name_all), 1)

        Es = np.zeros([len(sen_para), NDA, 2])  # Estimation results
        Err_r = np.zeros([len(sen_para), NDA])
        Err_d = np.zeros([len(sen_para), NDA])
        Proportion_Range_Predict_Array = np.zeros([len(sen_para), 1])
        Proportion_Depth_Predict_Array = np.zeros([len(sen_para), 1])
        # Testing
        for i_para in range(len(sen_para)):
            batch_size = NDA
            batch_ndx = NDA / batch_size
            MAPE_R = 0
            MAPE_Z = 0
            MAE_R = 0
            MAE_Z = 0
            test_set = SnSpectrumLoader(file_path=path, length_freq=length_freq, Sr=Sr, Sd=Sd, SNR=SNR,
                                        i_file=i_para, run_mode='test',
                                        model=model_name, num_of_receiver=num_of_receiver)
            dataTest = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

            for iba, mydataTest in enumerate(dataTest):
                with torch.no_grad():
                    inputs = mydataTest['C'].float()
                    inputs = Variable(inputs)
                    inputs = inputs.to(default_device).type(default_type)

                    r = mydataTest['r'].float() / r_max
                    # print('r',r*r_max)
                    r = r.to(default_device).type(default_type)
                    z = mydataTest['z'].float() / z_max
                    # print('z', z * z_max)
                    z = z.to(default_device).type(default_type)

                    loss, log_vars, output = self.model(inputs, [r, z])

                    MAPE_R += float(((output[0] - r).abs() / r).sum(0) / batch_size)
                    MAPE_Z += float(((output[1] - z).abs() / z).sum(0) / batch_size)
                    MAE_R += float((output[0] - r).abs().sum(0) / batch_size * r_max)
                    MAE_Z += float((output[1] - z).abs().sum(0) / batch_size * z_max)

                    # Output test error
                    if iba % 5 == 4 or True:
                        print(iba + 1, batch_ndx, 'MAE_R: %.3fkm MAE_Z: %.3fm'
                              % (MAE_R / (iba + 1), MAE_Z / (iba + 1)))

                    Es[i_para, iba * batch_size: (iba + 1) * batch_size, 0] = (
                            output[0].to(torch.device('cpu')).type(default_type) * r_max).numpy()
                    Es[i_para, iba * batch_size: (iba + 1) * batch_size, 1] = (
                            output[1].to(torch.device('cpu')).type(default_type) * z_max).numpy()

            Err_r[i_para, :] = np.abs(Es[i_para, :, 0] - SrExpand)
            N_Range_Predict_Right = np.sum(Err_r[i_para, :] <SrExpand*ErrorLimit_Range)
            # N_Range_Predict_Right = np.sum(Err_r[i_para, :] < ErrorLimit_Range)
            Proportion_Range_Predict_Array[i_para] = N_Range_Predict_Right / NDA
            print('Proportion rightly of Range Predict: %2.1f%%' % (Proportion_Range_Predict_Array[i_para] * 100))

            Err_d[i_para, :] = np.abs(Es[i_para, :, 1] - SdExpand)
            N_Depth_Predict_Right = np.sum(Err_d[i_para, :] < ErrorLimit_Depth)
            Proportion_Depth_Predict_Array[i_para] = N_Depth_Predict_Right / NDA
            print('Proportion rightly of Depth Predict: %2.1f%%' % (Proportion_Depth_Predict_Array[i_para] * 100))

        # plot
        # save_file = './Results/Test_results/ssp_shift'
        filepath_figure_save_folder = os.path.abspath(self.args.save_file + '/c.test_results/figure')
        # save_name_figure_file = model_name + '_test_accuracy'
        save_name_figure_file = (str.split(model_path, '/')[-1])[:-4] + '_test_accuracy'
        save_path_figure_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.svg'
        if not os.path.exists(filepath_figure_save_folder):
            os.makedirs(filepath_figure_save_folder)
        # save_dict = save_file + '/' + model_path + '_Proportion.png'

        pl.rcParams['font.sans-serif'] = ['Simhei']
        dpi = 200
        pl.figure(dpi=dpi, figsize=(10, 4))
        pl.subplot(121)
        bp(sen_para, Proportion_Range_Predict_Array * 100, title='(a) Range estimation results',
           x_label='Parameter', y_label='Proportion (%)', color='b', line_style='-', marker='o')
        pl.subplot(122)
        bp(sen_para, Proportion_Depth_Predict_Array * 100, title='(b) Depth estimation results',
           x_label='Parameter', y_label='Proportion (%)', color='b', line_style='-', marker='o')
        pl.tight_layout()
        pl.savefig(save_path_figure_file, dpi=dpi, bbox_inches='tight')
        # pl.show()

        # data
        filepath_data_save_folder = os.path.abspath(self.args.save_file + '/c.test_results/data')
        save_name_data_file = save_name_figure_file
        save_path_data_file = filepath_data_save_folder + '/' + save_name_data_file + '.mat'
        if not os.path.exists(filepath_data_save_folder):
            os.makedirs(filepath_data_save_folder)
        sio.savemat(save_path_data_file,
                    {'model_name': model_name,
                     'para': sen_para, 'PRPA': Proportion_Range_Predict_Array,
                     'PDPA': Proportion_Depth_Predict_Array,
                     'SNR': SNR, 'Es': Es, 'Sr': Sr, 'Sd': Sd, 'Err_r': Err_r, 'Err_d': Err_d})

    def exp_deal(self, model_path, mode='exp'):
        self.ckp.write_log('--------Begin Deal with Measured Data-------\n')
        # Using gpu to accelerate devices
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("running at device %s" % default_device)
        default_type = torch.float32

        # Load data
        length_freq = self.args.length_freq
        num_of_receiver = self.args.num_of_receiver
        data_size_one_source = 2 * length_freq * num_of_receiver + 2
        batch_size = self.args.batch_size_exp
        # file_path = os.path.abspath('../1.1 Dataset Simulation/1.1.3 Dataset/1.1.3.3 Measured Set/'
        #                             + self.args.data_exp + '.sim')
        path = os.path.abspath(self.args.data_loader_path + '/(3)Dataset/D.MeasuredSet')
        file_name_all = glob.glob(path + '/*.sim')
        file_path = file_name_all[0]
        file_size = os.path.getsize(file_path)

        # Normalized
        r_max = 20
        z_max = 100

        # Importing Models
        model_name = self.args.model
        self.model.to(default_device).type(default_type)
        self.model.eval()
        # path_checkpoint = './weight_parameter/' + model_path + '.pth'
        path_checkpoint = model_path
        checkpoint = torch.load(path_checkpoint)
        self.model.load_state_dict(checkpoint['model'])  # -*- coding: utf-8 -*-
        batch_ndx = int(file_size / data_size_one_source / 4 / batch_size)
        print('batch_ndx, batch_size', batch_ndx, batch_size)
        # Initialization parameters
        MAPE_r = 0
        MAPE_d = 0
        MAE_r = 0
        MAE_d = 0
        Es = np.zeros([batch_ndx * batch_size, 2])
        Target = np.zeros([batch_ndx * batch_size, 2])
        exp_set = SnSpectrumLoader(file_path=file_path, length_freq=length_freq, run_mode=mode, model=model_name,
                                   num_of_receiver=num_of_receiver)
        dataTest = data.DataLoader(exp_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
        for iba, mydataTest in enumerate(dataTest):
            with torch.no_grad():
                #  Importing Test Sets
                inputs = mydataTest['C'].float()
                inputs = Variable(inputs)
                inputs = inputs.to(default_device).type(default_type)

                r = mydataTest['r'].float() / r_max
                r = r.to(default_device).type(default_type)
                z = mydataTest['z'].float() / z_max
                z = z.to(default_device).type(default_type)

                loss, log_vars, output = self.model(inputs, [r, z])

                # Calculation error
                MAPE_r += float(((output[0] - r).abs() / r).sum(0) / batch_size)
                MAPE_d += float(((output[1] - z).abs() / z).sum(0) / batch_size)
                MAE_r += float((output[0] - r).abs().sum(0) / batch_size * r_max)
                MAE_d += float((output[1] - z).abs().sum(0) / batch_size * z_max)

                if iba % 5 == 4 or True:
                    print(iba + 1, batch_ndx, '|MAE_R: %.3fkm MAE_Z: %.3fm'
                          % (MAE_r / (iba + 1), MAE_d / (iba + 1)))

                # Output estimation results
                Es[iba * batch_size: (iba + 1) * batch_size, 0] = (
                        output[0].to(torch.device('cpu')).type(default_type) * r_max).numpy()
                Es[iba * batch_size: (iba + 1) * batch_size, 1] = (
                        output[1].to(torch.device('cpu')).type(default_type) * z_max).numpy()
                Target[iba * batch_size: (iba + 1) * batch_size, 0] = (r.to(torch.device('cpu')).
                                                                       type(default_type) * r_max).numpy()
                Target[iba * batch_size: (iba + 1) * batch_size, 1] = (z.to(torch.device('cpu')).
                                                                       type(default_type) * z_max).numpy()

        x = torch.arange(1, batch_ndx * batch_size + 1, 1)  # 投弹序列数
        AEr = np.abs(Es[:, 0] - Target[:, 0])
        AEd = np.abs(Es[:, 1] - Target[:, 1])

        print(
            '---Current mode: %s \n ---MAE_r is: %.3fkm\n ---MAE_d is %.2fm: ' % (mode, np.mean(AEr), np.mean(AEd)))
        self.ckp.write_log(
            '---Current mode: {} ---MAE_r is: {:.3f}km ---MAE_d is: {:.2f}m.\n'.format(mode, np.mean(AEr),
                                                                                       np.mean(AEd)))
        # 画图
        filepath_figure_save_folder = os.path.abspath(self.args.save_file + '/d.exp_results/figure')
        save_name_figure_file = (str.split(model_path, '/')[-1])[:-4] + '_exp_estimation'
        # save_name_figure_file = model_name + '_exp_estimation'
        save_path_figure_file = filepath_figure_save_folder + '/' + save_name_figure_file + '.svg'
        # save_file = './Results/Measured_data_processing'
        # save_dict = save_file + '/' + model_path + '_EXP.png'
        if not os.path.exists(filepath_figure_save_folder):
            os.makedirs(filepath_figure_save_folder)
        dpi = 200
        pl.figure(dpi=dpi, figsize=(8, 8))
        pl.subplot(221)
        bp(x, Es[:, 0], color='b', line_style='-', legend_label='Estimation value', marker='s')
        bp(x, Target[:, 0], title='(a) Range estimation results', x_label='Sample index', y_label='Range (km)',
           color='r', line_style='--', legend_label='real', marker='o')

        pl.subplot(222)
        bp(x, Es[:, 1], color='b', line_style='', legend_label='Estimation value', marker='s')
        bp(x, Target[:, 1], title='(b) Depth estimation results', x_label='Sample index', y_label='Depth (m)',
           color='r', line_style='', legend_label='real', marker='o')

        pl.subplot(223)
        bp(x, AEr, title='(c) Range estimation error', x_label='Sample index',
           y_label='Absolute error in range estimation (km)', color='b', line_style='', legend_label='', marker='o')

        pl.subplot(224)
        bp(x, AEd, title='(d) Depth estimation error', x_label='Sample index',
           y_label='Absolute error in depth estimation (m)', color='b', line_style='', legend_label='', marker='o')
        pl.tight_layout()
        pl.savefig(save_path_figure_file, dpi=dpi, bbox_inches='tight')
        # pl.show()

        #   Save results
        filepath_data_save_folder = os.path.abspath(self.args.save_file + '/d.exp_results/data')
        if not os.path.exists(filepath_data_save_folder):
            os.makedirs(filepath_data_save_folder)
        save_name_data_file = save_name_figure_file
        save_path_data_file = filepath_data_save_folder + '/' + save_name_data_file + '.mat'
        sio.savemat(save_path_data_file,
                    {'model_name': model_name,
                     'Es': Es, 'Sr': Target[:, 0],
                     'Sd': Target[:, 1],
                     'Err_r': AEr, 'Err_d': AEd})

    '''def plotloss(self, model_path):
        # pl.rcParams['font.sans-serif'] = 'Times New Roman'
        path_checkpoint = './权重文件/' + model_path + '.pth'  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        self.model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数

        # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        epoch_start = checkpoint['iter_num'] + 1  # 设置开始的epoch
        loss_epoch = checkpoint['loss_epoch']
        logr_list = checkpoint['logr_list']
        logz_list = checkpoint['logz_list']
        MAPE_R_list = checkpoint['MAPE_R_list']
        MAPE_Z_list = checkpoint['MAPE_Z_list']
        MAE_R_list = checkpoint['MAE_R_list']
        MAE_Z_list = checkpoint['MAE_Z_list']
        loss_epoch_v = checkpoint['loss_epoch_v']
        MAPE_R_list_v = checkpoint['MAPE_R_list_v']
        MAPE_Z_list_v = checkpoint['MAPE_Z_list_v']
        MAE_R_list_v = checkpoint['MAE_R_list_v']
        MAE_Z_list_v = checkpoint['MAE_Z_list_v']

        MAE_R_EXP_list = checkpoint['MAE_R_EXP_list']
        MAE_Z_EXP_list = checkpoint['MAE_Z_EXP_list']
        # epoch_start = len(loss_epoch)
        # 画图
        save_file = './结果/训练结果/图片'
        save_dict = save_file + '/' + model_path + '.svg'
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        sigmar = 10 ** np.array(logr_list)
        sigmaz = 10 ** np.array(logz_list)

        dpi = 200
        x = torch.arange(1, len(loss_epoch) + 1, 1) * self.args.NE
        pl.figure(dpi=dpi, figsize=(10, 3))
        pl.subplot(131)
        bp(x, MAE_R_list, x_tick_off=False, legend_label='Train', color='b')
        bp(x, MAE_R_list_v[self.args.NST:], title='', x_label='epoch',
           y_label=r'$MAE_r$ (km)', x_tick_off=False, legend_label='Vali', color='r',
           font_size=20, loc='upper right')
        pl.ylim([0, 6])
        pl.xlim([0, 3])
        pl.subplot(132)
        bp(x, MAE_Z_list, x_tick_off=False, legend_label='Train', color='b')
        bp(x, MAE_Z_list_v, title='', x_label='epoch',
           y_label=r'$MAE_z$ (m)', x_tick_off=False, legend_label='Vali', color='r'
           , font_size=20, loc='upper right')
        pl.yticks(ticks=[0, 10, 20, 30])
        pl.ylim([0, 30])
        pl.xlim([0, 3])
        pl.subplot(133)
        bp(x, sigmar, legend_label=r'$\sigma_r$')
        bp(x, sigmaz, line_style='--', color='r', title='', x_label='epoch',
           y_label=r'$\sigma$', x_tick_off=False, legend_label=r'$\sigma_z$', font_size=20, loc='upper right')
        pl.ylim([-0.2, 6])
        pl.xlim([0, 3])
        pl.tight_layout()
        pl.savefig(save_dict, dpi=dpi, bbox_inches='tight')
        pl.show()

        save_file = './结果/训练结果/数据'
        save_dict = save_file + '/' + model_path + '.png'
        if not os.path.exists(save_file):
            os.makedirs(save_file)

        if 'cnn' in model_path:
            sio.savemat(save_dict[:-3] + 'mat',
                        {'epoch': x.numpy(), 'CNN_MAE_R_TRA': np.array(MAE_R_list),
                         'CNN_MAE_Z_TRA': np.array(MAE_Z_list),
                         'CNN_MAE_R_Vali': np.array(MAE_R_list_v),
                         'CNN_MAE_Z_Vali': np.array(MAE_Z_list_v),
                         })
        else:
            sio.savemat(save_dict[:-3] + 'mat',
                        {'epoch': x.numpy(), 'UNet_MAE_R_TRA': np.array(MAE_R_list),
                         'UNet_MAE_Z_TRA': np.array(MAE_Z_list),
                         'UNet_MAE_R_Vali': np.array(MAE_R_list_v),
                         'UNet_MAE_Z_Vali': np.array(MAE_Z_list_v),
                         })
        
        
        ('------- Finished Training! -------')

        save_file = './结果/实验训练结果/图片'
        save_dict = save_file + '/' + model_path + '.png'
        if not os.path.exists(save_file):
            os.makedirs(save_file)

        # 画实验数据误差
        pl.figure(dpi=dpi, figsize=(12, 5))
        pl.subplot(121)
        bp(x, MAE_R_EXP_list, line_style='', marker='o', title='', x_label='训练周期', y_label='距离定位误差(km)')

        pl.subplot(122)
        bp(x, MAE_Z_EXP_list, line_style='', marker='o', title='', x_label='训练周期', y_label='深度定位误差(m)')
        pl.tight_layout()
        pl.savefig(save_dict, dpi=dpi, bbox_inches='tight')
        pl.show()

        save_file = './结果/实验训练结果/数据'
        save_dict = save_file + '/' + model_path + '.png'
        if not os.path.exists(save_file):
            os.makedirs(save_file)

        if 'cnn' in model_path:
            sio.savemat(save_dict[:-3] + 'mat',
                        {'epoch': x.numpy(), 'CNN_MAE_R_EXP': np.array(MAE_R_EXP_list),
                         'CNN_MAE_Z_EXP': np.array(MAE_Z_EXP_list)})
        else:
            sio.savemat(save_dict[:-3] + 'mat',
                        {'epoch': x.numpy(), 'UNet_MAE_R_EXP': np.array(MAE_R_EXP_list),
                         'UNet_MAE_Z_EXP': np.array(MAE_Z_EXP_list)})
        print('------- Finished Training! -------')
    '''
