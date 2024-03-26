# Start training
for current_epoch in tqdm_gui(range(start_epoch, max_epoch * mini_epoch)):

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
        train_set = SnSpectrumLoader(file_path=path, length_freq=num_of_frequency,
                                     SNR_range=SNR_range_of_train,
                                     run_mode=mode, model=model_name, num_of_receiver=num_of_receiver)
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
                    tqdm.write('|WARNING: run out of memory')
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
                tqdm.write('{:d} {:d} Train Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m'.
                           format(batch_idx + 1, batch_ndx, loss / (batch_idx + 1),
                                  MAE_r / (batch_idx + 1), MAE_d / (batch_idx + 1)))

                tqdm.write('Validation Loss: {:.3f} |MAE_R: {:.3f}km MAE_Z: {:.3f}m\n'.
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