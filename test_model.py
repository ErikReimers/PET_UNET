import argparse
import numpy as np
import numpy.matlib
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from model import get_model
import glob
import os
from PIL import Image
import tempfile
import imutils
from scipy.ndimage import gaussian_filter
from natsort import natsorted
from scipy.io import savemat



#Get inputs from user
def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--volume_path", type=str, required=True,
                        help="test volume dir")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_folder", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="if set, will display the ground truth image")
    parser.add_argument("--view", type=str, default="a",
                        help="which view do you want a = axial, c = coronal, s = sagittal,save = save full volume")
    parser.add_argument("--slice_nb", type=int, default=120,
                        help="which slice # do you want?")
    parser.add_argument("--degrees",type=int, default=0,
                        help="do you want to add rotations to the viewing?")
    parser.add_argument("--weights_nb",type=int, default=-1,
                        help="which weights file do you want to use, default will use the most recently saved")
    parser.add_argument("--volume_shape", nargs="*", type=int, default=[207,256,256],
                        help="What is the shape of volume? Default: 207 256 256")
    parser.add_argument("--sum_all", type=str, default="False",
                        help="Are you running on an entire folder and want to view the sum")
    parser.add_argument("--custom_max", type=float, default=0.,
                        help="Instead of loading in all the volumes to find the max value, just enter it yourself")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    volume_path = args.volume_path
    weight_folder = args.weight_folder
    ground_truth = args.ground_truth
    view = args.view
    slice_nb = args.slice_nb
    degrees = args.degrees
    weights_nb = args.weights_nb
    sum_all = args.sum_all
    custom_max = args.custom_max
    # 0.12993647158145905
    volume_shape = args.volume_shape
    #(89,276,276)
    #(53,256,256)
    #(207,256,256)

    #If unspecified, take the most recently saved weight file in the folder
    if weights_nb == -1:
       list_of_files = glob.glob(weight_folder + '/*.hdf5')
       weight_file = max(list_of_files, key=os.path.getctime)
    #Otherwise take the specified int value in the list of weight files
    else:
       list_of_files = natsorted(glob.glob(weight_folder + '/*.hdf5'),key=str)
       weight_file = list_of_files[weights_nb]

    #Load up the saved model
    model = get_model(args.model)
    model.load_weights(weight_file)
    
    #if saving the images, make the specified folder
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    #Check if the specified volume path is for a specific volume (.i) or a directory
    if volume_path[-1]=='i':  
        volume_paths = [volume_path]
    elif os.path.isdir(volume_path):  
        volume_suffixes = (".i")
        volume_paths = natsorted([p for p in Path(volume_path).glob("**/*") if p.suffix.lower() in volume_suffixes])

    nb_volumes = len(volume_paths)
    
    #Figure out if cropping needs to be done. UNET requires image shape to be multiply of 16
    crop_x = volume_shape[1]//16*16
    crop_y = volume_shape[2]//16*16
    new_volume_shape = [volume_shape[0],crop_x,crop_y]

    #Initialize the variables and determine shape of images
    #a: axial slice
    #c: coronal slice
    #s: sagittal slice
    #save: full 3D volume to be saved as .mat

    if view == "a":
        h = new_volume_shape[1]
        w = new_volume_shape[2]
        all_noisy_data = np.zeros((h,w,nb_volumes))
        all_denoised_data = np.zeros((h,w,nb_volumes))
        all_gt_data = np.zeros((h,w,nb_volumes))

    if view == "c":
        h = new_volume_shape[0]
        w = new_volume_shape[2]
        all_noisy_data = np.zeros((h,w,nb_volumes))
        all_denoised_data = np.zeros((h,w,nb_volumes))
        all_gt_data = np.zeros((h,w,nb_volumes))

    if view == "s":
        h = new_volume_shape[0]
        w = new_volume_shape[2]
        all_noisy_data = np.zeros((h,w,nb_volumes))
        all_denoised_data = np.zeros((h,w,nb_volumes))
        all_gt_data = np.zeros((h,w,nb_volumes))
    if view == "save":
        h = new_volume_shape[1]
        w = new_volume_shape[2]
        d = new_volume_shape[0]
        all_noisy_data = np.zeros((d,h,w,nb_volumes))
        all_denoised_data = np.zeros((d,h,w,nb_volumes))
        all_gt_data = np.zeros((d,h,w,nb_volumes))


    noise_volumes = np.zeros((new_volume_shape[0],new_volume_shape[1],new_volume_shape[2], nb_volumes))
    ground_truth_volumes = np.zeros((new_volume_shape[0],new_volume_shape[1],new_volume_shape[2], nb_volumes))
    #Open each volume file
    for jj in range(nb_volumes):

        ##Open, reshape and crop volume
        fid = open(volume_paths[jj], "r")
        noise_volume = np.reshape(np.fromfile(fid, dtype=np.float32), volume_shape)
        fid.close()
        
        noise_volumes[:,:,:,jj] = noise_volume[:,0:crop_x,0:crop_y]

        #If a specific ground truth file was given, use that
        if args.ground_truth:
            #Open, reshape and crop volume
            fid = open(args.ground_truth, "r")
            ground_truth_volume = np.reshape(np.fromfile(fid, dtype=np.float32), volume_shape)
            fid.close()

            ground_truth_volumes[:,:,:,jj] = ground_truth_volume[:,0:crop_x,0:crop_y]
        #Otherwise take a 3D guassian of the noise and call that the ground truth
        else:
            ground_truth_volumes[:,:,:,jj] = gaussian_filter(noise_volumes[:,:,:,jj],1)

    #Normalize the data, if there is a custom value, use that
    if custom_max != 0:
        noise_volumes = noise_volumes/custom_max
    else:
        print("max value of all volumes: ", np.max(noise_volumes))
        ground_truth_volumes = ground_truth_volumes/np.max(ground_truth_volumes)
        noise_volumes = noise_volumes/np.max(noise_volumes)
        

    for jj in range(nb_volumes):
        #if testing an axial view
        if view == "a":

            ground_truth_image = np.squeeze(ground_truth_volumes[slice_nb,:,:,jj])
            noise_image = np.squeeze(noise_volumes[slice_nb,:,:,jj])
            
            #If degrees were specified, rotate the noise image and the ground truth
            if degrees != 0:
                noise_image = imutils.rotate(noise_image,angle=degrees)
                ground_truth_image = imutils.rotate(ground_truth_image,angle=degrees)

            #Use the model to give a denoised image
            pred = model.predict(np.expand_dims(noise_image, 0))
            denoised_image = np.squeeze(pred[0])

            #Save the results to the all data variables
            all_noisy_data[:,:,jj] = noise_image
            all_denoised_data[:,:,jj] = denoised_image
            all_gt_data[:,:,jj] = ground_truth_image


        #If testing a coronal view
        elif view == "c":

            denoised_volume = np.zeros(new_volume_shape, dtype=np.float64)
            for ii in range(new_volume_shape[0]):
                slice_image = np.squeeze(noise_volumes[ii,:,:,jj])
                pred = model.predict(np.expand_dims(slice_image, 0))
                denoised_volume[ii,:,:] = np.squeeze(pred[0])

            noise_image = np.squeeze(noise_volumes[:,slice_nb,:,jj])
            denoised_image = np.squeeze(denoised_volume[:,slice_nb,:])
            ground_truth_image = np.squeeze(ground_truth_volumes[:,slice_nb,:,jj])

            #Save the results to the all data variables
            all_noisy_data[:,:,jj] = noise_image
            all_denoised_data[:,:,jj] = denoised_image
            all_gt_data[:,:,jj] = ground_truth_image

        #If testing a sagittal view
        elif view == "s":   

            denoised_volume = np.zeros(new_volume_shape, dtype=np.float64)
        
            for ii in range(new_volume_shape[0]):
                slice_image = np.squeeze(noise_volumes[ii,:,:,jj])
                pred = model.predict(np.expand_dims(slice_image, 0))
                denoised_volume[ii,:,:] = np.squeeze(pred[0])
        
            noise_image = np.squeeze(noise_volumes[:,:,slice_nb,jj])
            denoised_image = np.squeeze(denoised_volume[:,:,slice_nb])
            ground_truth_image = np.squeeze(ground_truth_volumes[:,:,slice_nb,jj])

            #Save the results to the all data variables
            all_noisy_data[:,:,jj] = noise_image
            all_denoised_data[:,:,jj] = denoised_image
            all_gt_data[:,:,jj] = ground_truth_image

        #If wanting to save the entire denoised volume
        elif view == "save":   

            denoised_volume = np.zeros(new_volume_shape, dtype=np.float64)
        
            for ii in range(new_volume_shape[0]):
                slice_image = np.squeeze(noise_volumes[ii,:,:,jj])
                pred = model.predict(np.expand_dims(slice_image, 0))
                denoised_volume[ii,:,:] = np.squeeze(pred[0])

            #Save the results to the all data variables
            all_noisy_data[:,:,:,jj] = np.squeeze(noise_volumes[:,:,:,jj])
            all_denoised_data[:,:,:,jj] = np.squeeze(denoised_volume)
            all_gt_data[:,:,:,jj] = np.squeeze(ground_truth_volumes[:,:,:,jj])

    if sum_all == "True":
        all_noisy_data = np.expand_dims(np.sum(all_noisy_data,axis=-1),-1)
        all_denoised_data = np.expand_dims(np.sum(all_denoised_data,axis=-1),-1)
        all_gt_data = np.expand_dims(np.sum(all_gt_data,axis=-1),-1)


    #Run through each file
    for jj in range(all_noisy_data.shape[2]):
        if view != "save":
            noise_image = np.squeeze(all_noisy_data[:,:,jj])
            denoised_image = np.squeeze(all_denoised_data[:,:,jj])
            ground_truth_image = np.squeeze(all_gt_data[:,:,jj])
            #Normalize the images according to their max pixel
            noise_image = noise_image/np.amax(all_noisy_data)
            denoised_image = denoised_image/np.amax(all_denoised_data)*2
            ground_truth_image = ground_truth_image/np.amax(all_gt_data)
            
            #Put all three images beside eachother for easy viewing
            out_image = np.zeros((h, w * 3), dtype=np.float64)

            out_image[:, :w] = ground_truth_image
            out_image[:, w:w * 2] = noise_image
            out_image[:, w * 2:] = denoised_image

            #if saving to files, also save small blown-up example squares
            if args.output_dir:
                plt.imsave(str(output_dir.joinpath(os.path.basename(volume_paths[jj])))[:-4] + ".png", np.squeeze(out_image), vmin=0, vmax=1)

                example1 = np.repeat(np.repeat(np.squeeze(out_image[96:127,111:148]),15,axis=0),15,axis=1)
                plt.imsave(str(output_dir.joinpath(os.path.basename(volume_paths[jj])))[:-4] + "zoomGT.png", example1, vmin=0, vmax=1) 
                example2 = np.repeat(np.repeat(np.squeeze(out_image[96:127,(111+w):(148+h)]),15,axis=0),15,axis=1)
                plt.imsave(str(output_dir.joinpath(os.path.basename(volume_paths[jj])))[:-4] + "zoomN.png", example2, vmin=0, vmax=1)
                example3 = np.repeat(np.repeat(np.squeeze(out_image[96:127,(111+w*2):(148+h*2)]),15,axis=0),15,axis=1)
                plt.imsave(str(output_dir.joinpath(os.path.basename(volume_paths[jj])))[:-4] + "zoomDN.png", example3, vmin=0, vmax=1) 
            #otherwise display a fullscreen plot
            else:
                plt.imshow(out_image, vmin=0, vmax=1)
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                plt.show()
            

        elif view == "save": 
            savemat(volume_path[0:-2]+"_denoised.mat", {"all_denoised_data": all_denoised_data, "label": "all_denoised_data"})
            savemat(volume_path[0:-2]+"_ground_truth.mat", {"all_gt_data": all_gt_data.astype(np.float64), "label": "all_gt_data"})
            savemat(volume_path[0:-2]+"_noisy.mat", {"noise_volume": all_noisy_data.astype(np.float64), "label": "all_noisy_data"})
        
        

if __name__ == '__main__':
    main()
