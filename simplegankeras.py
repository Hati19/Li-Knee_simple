from __future__ import print_function
from PIL import Image, ImageEnhance
import os
import sys
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import objectives
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import Iterator
import matplotlib.pyplot as plt
#plt.switch_backend('agg') # Very Important in R Markdown
import tensorflow as tf
import csv
from skimage import filters
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import measure
from sklearn.utils import shuffle
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

###############################################
img_ch=1 # image channels
out_ch=1 # output channel
seg_ch=1
# training settings
img_size= (384,384)
n_rounds=10
batch_size=10
n_filters_d=32
n_filters_g=32
val_ratio=0.2
init_lr=2e-4
nb_epoch=50000
alpha_recip=0.1
model_out_dir='weights/'
smooth=1
input_training_dir='/home/gajs/Downloads/Sibaji/OAI_iMorphicsSegmentation/train_all/'
#################################################
def discriminator_image(img_size, n_filters, init_lr, name='d'):
    """
    discriminator network (patch GAN)
      stride 2 conv X 2
        max pooling X 4
    fully connected X 1
    """

    # set image specifics
    k=3 # kernel size
    s=2 # stride

    img_height, img_width = img_size[0], img_size[1]
    padding='same'#'valid'

    inputs = Input((img_height, img_width, img_ch+seg_ch))

    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, kernel_size=(k, k), padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), strides=(s,s), padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*n_filters, kernel_size=(k, k), padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4*n_filters, kernel_size=(k, k), padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8*n_filters, kernel_size=(k, k), padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16*n_filters, kernel_size=(k, k), padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    gap=GlobalAveragePooling2D()(conv5)
    outputs=Dense(1, activation='sigmoid')(gap)

    d = Model(inputs, outputs, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                            K.batch_flatten(y_pred))
#         L = objectives.mean_squared_error(K.batch_flatten(y_true),
#                                            K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=d_loss, metrics=['accuracy'])

    return d, d.layers[-1].output_shape[1:]

def generator(img_size, n_filters, name='g'):
    """
    generate network based on unet
    """

    # set image specifics
    k=3 # kernel size
    s=2 # stride

    img_height, img_width = img_size[0], img_size[1]
    padding='same'

    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)

    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)

    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)

    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)

    g = Model(inputs, outputs, name=name)

    return g

def GAN(g,d,img_size,n_filters_g, n_filters_d, alpha_recip, init_lr, name='gan'):
    """
    GAN (that binds generator and discriminator)
    """
    img_h, img_w=img_size[0], img_size[1]


    image = Input((img_h, img_w, img_ch))
    mask = Input((img_h, img_w, seg_ch))

    fake_mask=g(image)
    fake_pair=Concatenate(axis=3)([image, fake_mask])

    gan=Model([image, mask], d(fake_pair), name=name)

    def gan_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)
#         L_adv = objectives.mean_squared_error(y_true_flat, y_pred_flat)

        mask_flat = K.batch_flatten(mask)
        fake_mask_flat = K.batch_flatten(fake_mask)
        L_seg = objectives.binary_crossentropy(mask_flat, fake_mask_flat)
#         L_seg = objectives.mean_absolute_error(mask_flat, fake_mask_flat)

        return alpha_recip*L_adv + L_seg


    gan.compile(optimizer=Adam(lr=init_lr, beta_1=0.5), loss=gan_loss, metrics=['accuracy'])

    return gan

def input2discriminator(real_img_patches, real_vessel_patches, fake_vessel_patches, d_out_shape):
    real=np.concatenate((real_img_patches,real_vessel_patches), axis=3)
    fake=np.concatenate((real_img_patches,fake_vessel_patches), axis=3)

    d_x_batch=np.concatenate((real,fake), axis=0)

    # real : 1, fake : 0
    #print(d_out_shape)
    d_y_batch=np.ones((d_x_batch.shape[0], d_out_shape[0]))
    d_y_batch[real.shape[0]:,...] = 0

    return d_x_batch, d_y_batch

def input2gan(real_img_patches, real_vessel_patches, d_out_shape):
    g_x_batch=[real_img_patches,real_vessel_patches]
    # set 1 to all labels (real : 1, fake : 0)
    g_y_batch=np.ones((real_vessel_patches.shape[0], d_out_shape[0]))
    return g_x_batch, g_y_batch

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def print_metrics(itr, **kargs):
    print ("*** Round {}  ====> ".format(itr),)
    for name, value in kargs.items():
        print (( "{} : {}, ".format(name, value)),end='')
    print ("")
    sys.stdout.flush()

def threshold_by_otsu(pred_vessels,  flatten=False):

    # cut by otsu threshold
    #threshold=filters.threshold_otsu(pred_vessels)
    threshold=0.9
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin


def dice_coefficient_in_train(true_vessel_arr, pred_vessel_arr):
    true_vessel_arr = true_vessel_arr.astype(np.bool)
    pred_vessel_arr = pred_vessel_arr.astype(np.bool)

    intersection = np.count_nonzero(true_vessel_arr & pred_vessel_arr)

    size1 = np.count_nonzero(true_vessel_arr)
    size2 = np.count_nonzero(pred_vessel_arr)

    try:
        dc = 2. * intersection / float(size1 + size2)
    except ZeroDivisionError:
        dc = 0.0

    return dc
def pixel_values_in_mask(true_vessels, pred_vessels, split_by_img=False):
    assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
    assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0
    #assert np.max(masks)==1.0 and np.min(masks)==0.0
    assert pred_vessels.shape[0]==true_vessels.shape[0] #and masks.shape[0]==true_vessels.shape[0]
    assert pred_vessels.shape[1]==true_vessels.shape[1] #and masks.shape[1]==true_vessels.shape[1]
    assert pred_vessels.shape[2]==true_vessels.shape[2] #and masks.shape[2]==true_vessels.shape[2]

    if split_by_img:
        n=pred_vessels.shape[0]
       # return np.array([true_vessels[i,...].flatten() for i in range(n)]), np.array([pred_vessels[i,...][masks[i,...]==1].flatten() for i in range(n)])
    else:
        return true_vessels.flatten(), pred_vessels.flatten()

def get_train_file_names(input_file_name):
    images = sorted(os.listdir(input_file_name))
    train_id=[]
    train_mask_id=[]
    for image_name in images:
        if 'mask'  in image_name:
            #print(os.path.join(input_file_name, image_name))
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        train_id.append(os.path.join(input_file_name, image_name))
        #print(os.path.join(input_file_name, image_name))
        train_mask_id.append(os.path.join(input_file_name, image_mask_name))


    return train_id,train_mask_id

def get_data(input_file_name_list):

    images=input_file_name_list
    total = len(images)
    imgs = np.ndarray((total, FLAGS.img_rows, FLAGS.img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, FLAGS.img_rows, FLAGS.img_cols), dtype=np.float32)
    i=0
    for image_name in images:

        if 'mask'  in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        i+=1
        #if(i%160)

    #print(np.amax(imgs))
    #print(np.amax(imgs_mask))
    return imgs, imgs_mask

def get_data_random_batch(input_file_name_list,batch_size):
    images=list(np.random.choice(input_file_name_list, batch_size))
    #images=input_file_name_list

    total = len(images)
    img_rows, img_cols=img_size[0], img_size[1]
    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    i=0
    for image_name in images:

        #if 'mask'  in image_name:
        #    continue

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        i+=1
        #if(i%160)

    #print(np.amax(imgs))
    #print(np.amax(imgs_mask))
    return imgs[..., np.newaxis], imgs_mask[..., np.newaxis]
def get_data_batch(input_file_name_list):
    #images=list(np.random.choice(input_file_name_list, batch_size))
    #images=input_file_name_list
    images=input_file_name_list
    total = len(images)
    img_rows, img_cols=img_size[0], img_size[1]
    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    i=0
    for image_name in images:

        #if 'mask'  in image_name:
        #    continue

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        i+=1
        #if(i%160)

    #print(np.amax(imgs))
    #print(np.amax(imgs_mask))
    return imgs[..., np.newaxis], imgs_mask[..., np.newaxis]
def get_data_binary_mask(input_file_name_list):

    images=input_file_name_list
    total = len(images)
    img_rows, img_cols=img_size[0], img_size[1]
    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total), dtype=np.float32)
    i=0
    for image_name in images:

        if 'mask'  in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        #img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        #img_mask = np.array([img_mask])
        imgs[i] = img
        #imgs_mask[i] = img_mask
        if np.sum(img_mask) != 0:
            #imgs.append(image_name)
            #print(image_name)
            imgs_mask[i] = 1
            #print(image_mask_name)
        else:
            imgs_mask[i] = 0
        i+=1
        #if(i%160)

    print(np.amax(imgs))
    print(np.amax(imgs_mask))
    return imgs[..., np.newaxis], imgs_mask

def create_binary_mask(input_file_name_list):

    images=input_file_name_list
    total = len(images)
    imgs_mask = np.ndarray((total), dtype=np.float32)
    i=0
    for image_name in images:

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        #img = np.load(image_name)
        #img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        #img_mask=np.squeeze(img_mask, axis=0)
        if np.sum(img_mask) != 0:
            #imgs.append(image_name)
            #print(image_name)
            imgs_mask[i] = 1
            #print(image_mask_name)
        else:
            imgs_mask[i] = 0
        i+=1

    #print(i)
    return imgs_mask

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def remove_balck_mask(input_file_name_list,total_mask_id):

    images=input_file_name_list
    total = len(images)

    #i=0
    imgs=[]
    imgs_mask=[]
    for image_name in images:

        #if 'mask'  in image_name:
        #    continue

        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        #img = np.load(image_name)
        #img=np.squeeze(img, axis=0)
        img_mask = np.load(image_mask_name)
        #img_mask=np.squeeze(img_mask, axis=0)
        if np.sum(img_mask) != 0:
            imgs.append(image_name)
            #print(image_name)
            imgs_mask.append(image_mask_name)
            #print(image_mask_name)
            #i+=1

    #print(i)
    return imgs, imgs_mask
###############################
# create networks
g = generator(img_size, n_filters_g)
d, d_out_shape = discriminator_image(img_size, n_filters_d,init_lr)
gan=GAN(g,d,img_size, n_filters_g, n_filters_d,alpha_recip, init_lr)

#g.summary()
#d.summary()
#gan.summary()

# start training data load
total_id,total_mask_id=get_train_file_names(input_training_dir)
#total_id=total_id[:160]
print(len(total_id))

total_id,total_mask_id=remove_balck_mask(total_id,total_mask_id)
#total_mask_id=create_binary_mask(total_id)
#print(total_mask_id)
#start_time = timeit.default_timer()
#total_id=total_id[1:100]
print(len(total_id))
np.random.shuffle(total_id)
train_id_temp=total_id[:int(len(total_id)*.9)]
test_id=total_id[int(len(total_id)*.9):]
with open('test_id.txt', 'w') as filehandle:
    for listitem in test_id:
        filehandle.write('%s\n' % listitem)

label=int((len(train_id_temp))*.7)
print(label)
train_id=train_id_temp[:label]
val_id=train_id_temp[label:]



print(len(train_id))
print(len(val_id))
print(len(test_id))


print('-'*30)
print('Fitting model...')
print('-'*30)
#model.fit(train_id, train_mask_id, batch_size=32, nb_epoch=100, verbose=1, shuffle=True,
#          validation_split=0.3,
#          callbacks=[model_checkpoint])
best_acc=0

for n_round in range(nb_epoch+1):

    # train D
    make_trainable(d, True)
    #print(n_round)
    for i in range(10):
        #print('In discriminator %d\n'%i)
        real_imgs, real_mask = get_data_random_batch(train_id,batch_size)
        d_x_batch, d_y_batch = input2discriminator(real_imgs, real_mask, g.predict(real_imgs), d_out_shape)
        loss, acc = d.train_on_batch(d_x_batch, d_y_batch)
        print_metrics(n_round+1, train_loss=loss, train_acc=acc, type='D')
        del d_x_batch
        del d_y_batch
        del real_imgs
        del real_mask

    # train G (freeze discriminator)
    make_trainable(d, False)
    for i in range(1):
        #print('In gan %d\n'%i)
        real_imgs, real_mask = get_data_random_batch(train_id,batch_size)
        g_x_batch, g_y_batch=input2gan(real_imgs, real_mask, d_out_shape)
        loss, acc = gan.train_on_batch(g_x_batch, g_y_batch)
        print_metrics(n_round+1, train_loss=loss, train_acc=acc, type='GAN')
        del g_x_batch
        del g_y_batch
        del real_imgs
        del real_mask
    if n_round%100 ==0: #goes for validation
        dice_coeff_total=0
        d_loss=0
        g_loss=0
        for i in range(0,len(val_id),batch_size):
            batch_id=train_id[i:min(i+batch_size,len(val_id))]
            real_imgs, real_mask = get_data_batch(batch_id)
            #print(real_imgs.shape)
            #print(real_mask.shape)
            pred_mask=g.predict(real_imgs)
            d_x_test, d_y_test=input2discriminator(real_imgs, real_mask, pred_mask, d_out_shape)
            loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
            #d_loss.append(loss)
            d_loss=d_loss+loss
            del d_x_test
            del d_y_test
            #print_metrics(i , Val_loss=loss, Val_acc=acc, type='D')
            # G
            gan_x_test, gan_y_test=input2gan(real_imgs, real_mask, d_out_shape)
            loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
            #g_loss.append(loss)
            g_loss=g_loss+loss
            #print_metrics(i, Val_loss=loss, Val_acc=acc,  type='GAN')
            #dice_coeff1=K.eval(dice_coef(real_mask, pred_mask))
            dice_coeff1=(np.sum(pred_mask*real_mask)*2.0 +1.0)/ (np.sum(pred_mask) + np.sum(real_mask)+1)
            dice_coeff_total=dice_coeff_total+dice_coeff1
            #print('dice_coeff %f'%dice_coeff1)
            #dice_coeff_total.append(dice_coeff1)
            #dice_coeff_total=dice_coeff_total+dice_coeff1
            del gan_x_test
            del gan_y_test
            del real_imgs
            del real_mask

        i=i/10
        #print_metrics(n_round+1, Val_loss=np.mean(d_loss),  type='D')
        #print_metrics(n_round+1, Val_loss=np.mean(g_loss),  type='D')
        #print_metrics(n_round+1, dice_coeff_total=np.sum(dice_coeff_total), dice_coeff_mean=np.mean(dice_coeff_total))
        print_metrics(n_round+1, Val_loss=d_loss/i,  type='D')
        print_metrics(n_round+1, Val_loss=g_loss/i,  type='GAN')
        print_metrics(n_round+1, dice_coeff_total=dice_coeff_total, dice_coeff_mean=dice_coeff_total/i)
            # save the weights
            #g.save_weights(os.path.join(model_out_dir,"g_{}_{}_{}.h5".format(n_round,discriminator,ratio_gan2seg)))
        dice_coeff_mean=dice_coeff_total/i
        if best_acc < dice_coeff_mean :
            print('Save Model')
            best_acc=dice_coeff_mean
            g.save_weights(os.path.join(model_out_dir,"g_{}_{:.3f}.h5".format(n_round,dice_coeff_mean)))
            if (best_acc > .7):
                dice_coeff_total=0
                d_loss=0
                g_loss=0
                for i in range(0,len(test_id),batch_size):
                    batch_id=train_id[i:min(i+batch_size,len(test_id))]
                    real_imgs, real_mask = get_data_batch(batch_id)
                    #print(real_imgs.shape)
                    #print(real_mask.shape)
                    pred_mask=g.predict(real_imgs)
                    d_x_test, d_y_test=input2discriminator(real_imgs, real_mask, pred_mask, d_out_shape)
                    loss, acc=d.evaluate(d_x_test,d_y_test, batch_size=batch_size, verbose=0)
                    #d_loss.append(loss)
                    d_loss=d_loss+loss
                    del d_x_test
                    del d_y_test
                    #print_metrics(i , Val_loss=loss, Val_acc=acc, type='D')
                    # G
                    gan_x_test, gan_y_test=input2gan(real_imgs, real_mask, d_out_shape)
                    loss,acc=gan.evaluate(gan_x_test,gan_y_test, batch_size=batch_size, verbose=0)
                    #g_loss.append(loss)
                    g_loss=g_loss+loss
                    #print_metrics(i, Val_loss=loss, Val_acc=acc,  type='GAN')
                    #dice_coeff1=K.eval(dice_coef(real_mask, pred_mask))
                    dice_coeff1=(np.sum(pred_mask*real_mask)*2.0 +1.e-10)/ (np.sum(pred_mask) + np.sum(real_mask)+1.e-10)
                    dice_coeff_total=dice_coeff_total+dice_coeff1
                    #print('dice_coeff %f'%dice_coeff1)
                    #dice_coeff_total.append(dice_coeff1)
                    #dice_coeff_total=dice_coeff_total+dice_coeff1
                    del gan_x_test
                    del gan_y_test
                    del real_imgs
                    del real_mask
                i=i/10
                print_metrics(n_round+1, test_loss=d_loss/i,  type='D')
                print_metrics(n_round+1, test_loss=g_loss/i,  type='GAN')
                print_metrics(n_round+1, dice_coeff_total=dice_coeff_total, dice_coeff_mean=dice_coeff_total/i)
