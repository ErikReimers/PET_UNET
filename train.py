import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model import get_model, PSNR
from generator import TrainingImageGenerator, ValGenerator

#This class decays the learning rate according the the % of epoches that have passed
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125

#Get the inputs from the user
def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=32,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=10,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', or 'mae', is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--rotations", type=str, default="False",
                        help="Do you want random rotations added, True or False")
    parser.add_argument("--sum_label", type=str, default="False",
                        help="Do you want the label to be a sum of the frames, True or False")
    parser.add_argument("--gt_dir", type=str, default="not_specified",
                        help="If you want to reference ground truth for validation, specify the folder")
    parser.add_argument("--nb_val_images", type=int, default=20,
                        help="how many validation images to include")
    parser.add_argument("--disp_examples", type=str, default="False",
                        help="display example patches before starting")
    parser.add_argument("--volume_shape", nargs="*", type=int, default=[207,256,256],
                        help="Are you running on Connor's phantom?")
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    image_dir = args.image_dir
    test_dir = args.test_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    sum_label = args.sum_label
    rotations = args.rotations
    gt_dir = args.gt_dir
    nb_val_images = args.nb_val_images
    disp_examples = args.disp_examples
    volume_shape = args.volume_shape

    #Output_path
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)

    #Get the model from the model.py file
    model = get_model(args.model)

    #If the user specified weights then load those in as a starting point
    if args.weight is not None:
        model.load_weights(args.weight)

    opt = Adam(learning_rate=lr)
    callbacks = []

    #Compile the model and get the image pair generator classes
    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    generator = TrainingImageGenerator(image_dir, batch_size=batch_size, image_size=image_size,rotations=rotations,sum_label=sum_label,volume_shape=volume_shape)
    val_generator = ValGenerator(test_dir,gt_dir,nb_val_images = nb_val_images, rotations=rotations,sum_label=sum_label,volume_shape=volume_shape)

    #Display a quick example patch pair and validation pair to make sure it's what you want
    if disp_examples == "True":
        xg,yg=generator[0]
        for ii in range(3):
            xv,yv=val_generator[ii]
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(np.squeeze(xg[ii,:,:,:]), vmin=0, vmax=np.max(xg[ii,:,:,:]) if volume_shape[2] == 276 else np.max(xg[ii,:,:,:])*0.2)
            axarr[0,1].imshow(np.squeeze(yg[ii,:,:,:]), vmin=0, vmax=np.max(yg[ii,:,:,:]) if sum_label == "True" or volume_shape[2] == 276 else np.max(yg[ii,:,:,:])*0.2)
            axarr[1,0].imshow(np.squeeze(xv), vmin=0, vmax=np.max(xv) if volume_shape[2] == 276 else np.max(xv)*0.2)
            axarr[1,1].imshow(np.squeeze(yv), vmin=0, vmax=np.max(yv) if sum_label == "True" or gt_dir != "not_specified" or volume_shape[2] == 276 else np.max(yv)*0.2)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()


    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}.hdf5",#-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
    #                                 save_best_only=True))
                                     save_freq=int(10*steps)))

    hist = model.fit(generator,
                    steps_per_epoch=steps,
                    epochs=nb_epochs,
                    validation_data=val_generator,
                    verbose=1,
                    callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
