#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:15:27 2021

@author: matthew
"""


#%% Imports



import numpy as np
import numpy.ma as ma 
import matplotlib.pyplot as plt
import pdb
import pickle
import sys
import glob
import os
from pathlib import Path
import shutil
import time
from contextlib import redirect_stdout                                             # used to send model summary to a .txt file
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import losses, optimizers



def make_dem_crops(dem, lons_mg, lats_mg, n_files, n_per_file, n_pix, outdir = None):
    """Given a DEM (and the lons and lats of each pixel in it), create random crops of it.  
    Inputs:
        dem | masked array | DEM
        lons_mg | array | lons of each pixel
        lats_mg | array | lats of each pixel
        n_files | int | number of files to split the new DEMs over.  
        n_per_file | int | number of DEMs per file.  
        outdir | string/path | directory to save files to.  
    Returns:
        dems | list of dicts | each DEM is dict in a list, and each dict has various keys and variables, such as dem, lons_mg, lats_mg, name, and centre.  
        Files containing DEMs.
    History:
        2020_03_02 | MEG | Written.  
    """
    
    if type(n_pix) is not int:
        raise Exception(f"'n_pix' must be an integer, but it is currently {type(n_pix)}.  Exiting.  ")

    
    for file_n in range(n_files):
        dems = []                                                                                     # each (mini) DEM will be an item in the list
        for dem_n in range(n_per_file):
            start_x = np.random.randint(0, dem.shape[1]- n_pix)                                # generat the pixel number to start cropping from in x
            start_y = np.random.randint(0, dem.shape[0]- n_pix)                                # and in y
            dem_minis  = dem[start_y:(start_y+n_pix), start_x:(start_x+n_pix)]                 # make random crop and assign to array.  
            lons_minis = lons_mg[start_y:(start_y+n_pix), start_x:(start_x+n_pix)]                    # make random crop and assign to array.  
            lats_minis = lats_mg[start_y:(start_y+n_pix), start_x:(start_x+n_pix)]                    # make random crop and assign to array.  
          
            dems.append({'dem'     : dem_minis,
                         'lons_mg' : lons_minis,
                         'lats_mg' : lats_minis,
                         'name'    : f"iberia_dem_{dem_n}",
                         'centre'  : (lons_minis[int(n_pix/2),int(n_pix/2)],                            # centre pixel of scene, lon
                                      lats_minis[int(n_pix/2),int(n_pix/2)])})                          # and lat
            
            
        if outdir is not None:
            with open(Path(f'{outdir}/dem_file_{file_n}.pkl'), 'wb') as f:
                pickle.dump(dems, f)                                                    # usual output style is many channel formats in a dict, but we are only interesetd in the one we generated.  
            f.close()
        print(f"Completed file {file_n} of {n_files}.")
        
    return dems

  


def file_merger(files): 
    """Given a list of files, open them and merge into one array.  
    Inputs:
        files | list | list of paths to the .npz files
    Returns
        X | r4 array | data
        Y_class | r2 array | class labels, ? x n_classes
        Y_loc | r2 array | locations of signals, ? x 4 (as x,y, width, heigh)
    History:
        2020/10/?? | MEG | Written
        2020/11/11 | MEG | Update to remove various input arguments
    
    """
    import numpy as np
    
    def open_synthetic_data_npz(name_with_path):
        """Open a file data file """  
        data = np.load(name_with_path)
        X = data['X']
        Y = data['Y']
        dem = data['dem']
        return X, Y, dem

    n_files = len(files)
    
    for i, file in enumerate(files):
        X_batch, Y_batch, dem_batch = open_synthetic_data_npz(file)
        if i == 0:
            n_data_per_file = X_batch.shape[0]
            X = np.zeros((n_data_per_file * n_files, X_batch.shape[1], X_batch.shape[2], X_batch.shape[3]))      # initate array, rank4 for image, get the size from the first file
            Y = np.zeros((n_data_per_file  * n_files, X_batch.shape[1], X_batch.shape[2], X_batch.shape[3]))                              # should be flexible with class labels or one hot encoding
            dem = np.zeros((n_data_per_file * n_files, X_batch.shape[1], X_batch.shape[2], X_batch.shape[3]))                                                     # four columns for bounding box
            
        
        X[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:,:,:] = X_batch
        Y[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:] = Y_batch
        dem[i*n_data_per_file:(i*n_data_per_file)+n_data_per_file,:] = dem_batch
    
    return X, Y, dem





def visualise_UnwrapNet(X, Y, model, n_data = 10):
    """Given some data (X), the labels (Y), and a model, predict the labels on n_data of these, 
    then plot the predictions (and the residuals)
    Inputs:
        X | rank 4 array | n_ims first, channels last.  Usually wrapped data.  
        Y | rank 4 array | n_ims first, channels last.  Usually unwrapped data
        model | keras model | used to predict Y_fcn
        n_data |  int | up to this many data will be plotted
    Returns:
        Matplotlib figure
    History:
        2021_03_03 | MEG | Written.  
    """
    
    import matplotlib.pyplot as plt
    
    if n_data > X.shape[0]:
        n_data = X.shape[0]

    Y_fcn = model.predict(X[:n_data,], verbose = 1)                                # forward pass of the testing data bottleneck features through the fully connected part of the model
    
    fig, axes = plt.subplots(4, n_data)  
    if n_data == 1:    
        axes = np.atleast_2d(axes).T                                                # make 2d, and a column (not a row)
    
    row_labels = ['wrapped (X)', 'Unw. (Y)', 'Unw. model(Y_fcn)', 'Resid.' ]
    for ax, label in zip(axes[:,0], row_labels):
        ax.set_ylabel(label)
    
    
    imshow_settings = {'interpolation' : 'none', 
                       'aspect'        : 'equal'}
    
    for data_n in range(n_data):
        
        # 0 calcuated the min and max values for the unwrapped data.  
        Y_combined = np.concatenate((Y[data_n,], Y_fcn[data_n]), axis = 0)   
        Y_min = np.min(Y_combined)
        Y_max = np.max(Y_combined)
        
        # 1: Do each of the 4 plots
        w = axes[0,data_n].imshow(X[data_n,:,:,0], **imshow_settings)                                                   # wrapped data
        axin = axes[0,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(w, cax=axin, orientation='horizontal')
        
        unw = axes[1,data_n].imshow(Y[data_n,:,:,0], **imshow_settings, vmin = Y_min, vmax = Y_max)                     # unwrapped 
        axin = axes[1,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(unw, cax=axin, orientation='horizontal')
        
        unw_cnn = axes[2,data_n].imshow(Y_fcn[data_n,:,:,0], **imshow_settings, vmin = Y_min, vmax = Y_max)             # unwrapped predicted by the model
        axin = axes[2,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(unw, cax=axin, orientation='horizontal')
        
        resid = axes[3,data_n].imshow((Y[data_n,:,:,0] - Y_fcn[data_n,:,:,0]), **imshow_settings)                       # difference betwee two unwrappeds.  
        axin = axes[3,data_n].inset_axes([0, -0.06, 1, 0.05])
        fig.colorbar(resid, cax=axin, orientation='horizontal')
    
    for ax in np.ravel(axes):
        ax.set_yticks([])
        ax.set_xticks([])
        

#%% 0: Things to set

dependency_paths = {'syinterferopy_bin'  : '/home/matthew/university_work/01_blind_signal_separation_python/SyInterferoPy/lib/',              # Available from Github: https://github.com/matthew-gaddes/SyInterferoPy
                    'srtm_dem_tools_bin' : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/',                                      # Available from Github: https://github.com/matthew-gaddes/SRTM-DEM-tools
                    'VUDL-NET-21'        : '/home/matthew/university_work/02_neural_networks_python/07_VUDL-Net-21',
                    'local_scripts'      : '/home/matthew/university_work/python_stuff/python_scripts/'}

dem_loc_size = {'west' : -8,
                'east' : -1,
                'south': 37,
                'north': 43}

dem_make = False
dem_settings = {"download"              : True,                                 # if don't need to download anymore, faster to set to false
                "void_fill"             : True,                                 # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "SRTM3_tiles_folder"    : './SRTM3/',                            # folder to keep SRTM3 tiles in 
                "water_mask_resolution" : 'i'}                                   # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)




# #step 02 (making synthetic interferograms):
ifg_settings            = {'n_per_file'         : 5}                                            # number of ifgs per data file.  
synthetic_ifgs_n_files  =  4                                                                    # numer of files of synthetic data
synthetic_ifgs_folder   = ''
synthetic_ifgs_settings = {'defo_sources'           : ['dyke', 'sill', 'no_def'],               # deformation patterns that will be included in the dataset.  
                            'n_ifgs'                 : ifg_settings['n_per_file'],               # the number of synthetic interferograms to generate PER FILE
                            'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                            'outputs'                : ['uuu', 'www', 'rid'],                                  # channel outputs.  uuu = unwrapped across all 3
                            'intermediate_figure'    : False,                                    # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                            'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                            'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                            'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                            'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                            'turb_aps_mean'          : 0.02}                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)


cnn_settings = {'n_files_train'    : 2,                                              # the number of files that will be used to train the network
                'n_files_validate' : 1,                                               # the number of files that wil be used to validate the network (i.e. passed through once per epoch)
                'n_files_test'     : 1}                                               # the number of files held back for testing.  

n_epochs = 5

   
np.random.seed(0)                                                                                           # 0 used in the example
              
#%% Import dependencies (paths set above)

for dependency_name, dependency_path in dependency_paths.items():
    sys.path.append(dependency_path)


from dem_tools_lib import SRTM_dem_make                                       # Used to make a DEM

from random_generation_functions import create_random_synthetic_ifgs
from auxiliary_functions import col_to_ma, griddata_plot, plot_ifgs                        

from detect_locate_nn_functions import file_list_divider

from small_plot_functions import matrix_show


n_pixs_dem = int(2 * synthetic_ifgs_settings['n_pix'])                                                                                           # the DEMs must be bigger than the ifgs that we'll make

#%% 1: Make a single master DEM
if dem_make:
    print(f"Making the DEM, which can be slow.  ")
    dem_settings['ed_username'] = input(f'Please enter your USGS Earthdata username:  ')
    dem_settings['ed_password'] = input(f'Please enter your USGS Earthdata password (NB characters will be visible!   ):  ')
    
    dem_master, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
    griddata_plot(dem_master, lons_mg, lats_mg,  "05 back to the original, 20km scene")
    
    with open(Path(f'./02_dem_files/master_dem.pkl'), 'wb') as f:
        pickle.dump(dem_master, f)                                                    # usual output style is many channel formats in a dict, but we are only interesetd in the one we generated.  
        pickle.dump(lons_mg, f)
        pickle.dump(lats_mg, f)
    f.close()
else:
    print(f"Loading an existing DEM... ", end = '')
    with open(Path(f'./02_dem_files/master_dem.pkl'), 'rb') as f:
        dem_master = pickle.load(f)
        lons_mg = pickle.load(f)
        lats_mg = pickle.load(f)
    f.close()
    print(f"Done.")
        



#%% 3: Create or load the synthetic interferograms.  


n_synth_data = ifg_settings['n_per_file'] * synthetic_ifgs_n_files

for file_n in range(synthetic_ifgs_n_files):
    print(f"Generating file {file_n} of {synthetic_ifgs_n_files} files.  ")

    dems = make_dem_crops(dem_master, lons_mg, lats_mg, n_files = 1, n_per_file=ifg_settings['n_per_file'], n_pix = n_pixs_dem)       # create random crops of the DEM for using in synthetic ifgs.     
 
    X_all, Y_class, Y_loc, Y_source_kwargs = create_random_synthetic_ifgs(dems, **synthetic_ifgs_settings)                             # generate the synthetic ifgs for this file.  
    
    X = X_all['www'][:,:,:,0].filled(fill_value = 0)[:,:,:,np.newaxis]                                                                 # get the wrapped phase, and fill any masked pixels with 0s
    Y = X_all['uuu'][:,:,:,0].filled(fill_value = 0)[:,:,:,np.newaxis]                                                                 # get the unwrapped phase
    dem = X_all['rid'][:,:,:,2].filled(fill_value = ma.mean(X_all['rid'][:,:,:,2]))[:,:,:,np.newaxis]                                  # get hte dem, but this time fill with the mean value of the DEM (as it could be a long way from 0 in mountains)
    
    np.savez(f'./03_synthetic_ifgs/data_file_{file_n}.npz', X = X, Y = Y, dem=dem)                                                     # save the data, now as a .npz as we aren't using masked arrays.  
    
    del X_all, Y_class, Y_loc


#%% Train test split of files?

data_files = sorted(glob.glob(f'03_synthetic_ifgs/*npz'), key = os.path.getmtime)                  # make list of data files


if len(data_files) < (cnn_settings['n_files_train'] + cnn_settings['n_files_validate'] + cnn_settings['n_files_test']):
    raise Exception(f"There are {len(data_files)} data files, but {cnn_settings['n_files_train']} have been selected for training, "
                    f"{cnn_settings['n_files_validate']} for validation, and {cnn_settings['n_files_test']} for testing, "
                    f"which sums to greater than the number of data files.  Perhaps adjust the number of files used for the training stages? "
                    f"For now, exiting.")

data_files_train, data_files_validate, data_files_test = file_list_divider(data_files, cnn_settings['n_files_train'], cnn_settings['n_files_validate'], cnn_settings['n_files_test'])                              # divide the files into train, validate and test




#%% Train a model (add the DEM as an input?)



def train_unw_network(model, files, n_epochs, loss_names, X_validate, Y_validate, outdir):
    """Train a double headed model using training data stored in separate files.  
    Inputs:
        model | keras model | the model to be trained
        files | list | list of paths and filenames for the files used during training
        n_epochs | int | number of epochs to train for
        loss names | list | names of outputs of losses (e.g. "class_dense3_loss)
    Returns
        model | keras model | updated by the fit process
        metrics_loss | r2 array | columns are: total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
        metrics_class | r2 array | columns are class accuracy, validation class accuracy
        
    2019/03/25 | Written.  
    2021/03/08 | MEG | Save the model after each epoch.  A crude approach to early stopping
    """
    import numpy as np
    import keras
    
    def plot_single_train_validate_loss(metrics_loss, n_files, outdir, n_epoch,
                                        pointsize = 2, spacing = 2):
        """ Create a plot showing training and validation loss when training using data split across multiple files.  
        Inputs:
            metrics_loss | numpy array | (n_files * n_epochs) x 2, becuase there is a loss for each file in every epoch.  First column for training, second for validation (so mostly 0s as only 1 pass per epoch)
            n_files | int | number of files being passed
            out_dir | string or Path | directory to save the figure to
            n_epoch | which epoch number currently up to (i.e epoch 5 of 10)
            pontsize | int | size of points in plot
            spacing | int | every nth epoch is labelled on the x axis.  
        """
        
        
        all_epochs = np.arange(0, metrics_loss.shape[0])                                         # n_files * n_epochs
        validation_epochs = np.arange(-1, metrics_loss.shape[0], n_files)[1:]                    # n_epochs (as only do the validation data once an epoch)
        
        f, ax = plt.subplots(1,1)
        ax.scatter(all_epochs[:n_files*(n_epoch+1)], metrics_loss[:,0][:n_files*(n_epoch+1)], s=pointsize )                                  # training loss
        ax.scatter(validation_epochs[:n_files*(n_epoch+1)], metrics_loss[validation_epochs,1][:n_files*(n_epoch+1)], s=pointsize )           # validation loss
        
        ax.grid(True, alpha = 0.2, which = 'both')
        ax.set_xlim([0, all_epochs.shape[0]])
        ax.set_ylim(bottom = 0)
        ax.set_xticks(np.arange(0,n_files * n_epochs,spacing * n_files))                 # change so a tick only after each epoch (and not each file)
        ax.set_xticklabels(np.arange(0,n_epochs, spacing))                                  # number ticks
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch n')
        f.savefig(Path(f"{outdir}/epoch_{n_epoch:003}_training_progress.png"), bbox_inches='tight')
        plt.close(f)

    
    n_files_train = len(files)                                                              # get the number of training files
    
    metrics_loss = np.zeros((n_files_train*n_epochs, 2))                                     # total loss/class loss/loc loss /validate total loss/validate class loss/ validate loc loss
    
    for e in range(n_epochs):                                                                        # loop through the number of epochs
        for file_num, file in enumerate(files):                                   # for each epoch, loop through all files once
        
            data = np.load(file)
            X_batch = data['X']
            Y_batch = data['Y']
            
            history_train_temp = model.fit(X_batch, Y_batch, batch_size=32, epochs=1, verbose = 0)                      # train it on one file
            metrics_loss[(e*n_files_train)+file_num, 0] = history_train_temp.history['loss'][0]                        # main loss    
            print(f'Epoch {e}, file {file_num}: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 0],0)} ')

            
        history_validate_temp = model.evaluate(X_validate, Y_validate, batch_size = 32, verbose = 0)                    # predict on validation data
        metrics_loss[(e*n_files_train)+file_num, 1] = history_validate_temp[0]                                          # main loss, validation
        print(f'Epoch {e}, valid.: Loss = {round(metrics_loss[(e*n_files_train)+file_num, 1],0)} ')
        
        print(f"Saving the current model...", end = '')
        model.save(Path(f"{outdir}/epoch_{e:03}_model_weights"))                                                        # save the model after each epoch
        print('Done.  ')
        
        #pdb.set_trace()
        plot_single_train_validate_loss(metrics_loss, len(files), outdir, e)

        
    
    return model, metrics_loss



model_output_dir = Path(f"./unwrap_net_run_{time.strftime('%Y_%m_%d_%H_%M_%S')}")              # make the name of a directory for this model run
os.makedirs(model_output_dir)                                                                  # make a folder to save things for this run


# 1 open the data that is not handled in a file by file manner
X_validate, Y_validate, dem_validate = file_merger(data_files_validate)                             # Open all the validation data to RAM
X_test, Y_test, dem_test = file_merger(data_files_test)                             # 

unwrap_FCN_input = keras.Input(shape=X_validate[0,].shape)                                                                                       # n_batch is ommited here
x = layers.Conv2D(filters = 64, kernel_size = 3, activation='relu', input_shape = X_validate[0,].shape, padding ='same')(unwrap_FCN_input)        # first layers of the model are 1D convolutions.  
x = layers.Conv2D(filters = 128, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 64, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 32, kernel_size = 3, activation='relu', padding ='same')(x)
x = layers.Conv2D(filters = 16, kernel_size = 3, activation='relu', padding ='same')(x)
unwrap_FCN_output = layers.Conv2D(filters = 1, kernel_size = 1, activation='relu', padding ='same')(x)


unwrap_FCN_model = keras.Model(inputs = unwrap_FCN_input, outputs = unwrap_FCN_output, name = 'unwrap_FCN')                                                      # build hte model
unwrap_FCN_model.compile(optimizers.Nadam(clipnorm = 1., clipvalue = 0.5), losses.mean_squared_error, metrics = [tf.keras.metrics.MeanSquaredError()])     # compile.  


with open(f"{model_output_dir}/model_summary.txt", 'w') as f:                                                                                            # summary of model to a text file.  
    with redirect_stdout(f):
        unwrap_FCN_model.summary()                                                                                                                                 # summary to terminal.  
keras.utils.plot_model(unwrap_FCN_model, f"{model_output_dir}/unwrap_FCN.png", show_shapes=True)                                                        # plot of the model to a png file.


unwrap_FCN_model, metrics_loss = train_unw_network(unwrap_FCN_model, data_files_train, n_epochs, ['loss'], X_validate, Y_validate, 
                                                   model_output_dir)                                                                                       # train

unwrap_FCN_model.save(f"{model_output_dir}/unwrap_FCN_model_final")

#%% Evaluate / predict


#reconstructed_model = keras.models.load_model("unwrap_FCN_model")
            

visualise_UnwrapNet(X_validate, Y_validate, unwrap_FCN_model, n_data = 10)

