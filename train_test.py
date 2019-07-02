# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:21:50 2019

@author: Sooram Kang

"""
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import load_data as ld
from cyclegan import img

class ImagePool(object):
    """
    keep 'maxsize' number of images to update the descriminator 
    """
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        # add fake images
        if self.num_img < self.maxsize: 
            self.images.append(image)
            self.num_img += 1
            return image
        # replace a random image with a new image if there are maxsize+ images  
        if np.random.rand() > 0.5:      
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image

def visualize(real, fake_A, fake_B, filename):
    """ 
    A1  A2  ... An 
    B1_ B2_ ... Bn_ (fake B made from A)
    B1  B2  ... Bn
    A1_ A2_ ... An_ (fake A made from B)
    
    """

    size = len(real) // 2
    A   = np.concatenate(real[:size], axis=1)
    B_  = np.concatenate(fake_B, axis=1)            
    B   = np.concatenate(real[size:], axis=1)
    A_  = np.concatenate(fake_A, axis=1)            
    result = np.concatenate([A, B_, B, A_], axis=0)
    
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(filename, dpi=256)
    
    
    
def train(args, model, sess):
#    summary_writer = tf.summary.FileWriter(model.log_dir, sess.graph)
    
    # load data
    filenames = ld.load_filenames(args.data_dir)
    
    # load previous model
    model.load(sess, args.model_dir)  
    
    pool = ImagePool(args.max_size)

    steps_per_epoch = (len(filenames[0])+len(filenames[1])) // (args.batch_size*2)        

    for epoch in range(args.start_epoch, args.epoch):
        if epoch < 100:
            curr_lr = args.d_lr
        else:
            curr_lr = args.d_lr - \
                args.d_lr * (epoch - 100) / 100

        for step in range(steps_per_epoch):
            x_batch = ld.get_random_batch(filenames, args.batch_size, img['w'])  

            # update G_AtoB network and record fake outputs
            fake_B, g_AtoB_loss, _ = sess.run([model.fake_B, model.loss_g_AtoB,  
                                               model.g_A_trainer],
                                 feed_dict={model.real_A: x_batch[:args.batch_size],
                                            model.real_B: x_batch[args.batch_size:],
                                            model.lr: curr_lr} )
            
            # update G_BtoA network and record fake outputs
            fake_A, g_BtoA_loss, _ = sess.run([model.fake_A, model.loss_g_BtoA, 
                                               model.g_B_trainer],
                                 feed_dict={model.real_A: x_batch[:args.batch_size],
                                            model.real_B: x_batch[args.batch_size:],
                                            model.lr: curr_lr})
            
            [fake_A, fake_B] = pool([fake_A, fake_B])
            
            # update D_A network
            d_A_loss, _ = sess.run([model.loss_d_A, model.d_A_trainer], 
                     feed_dict={model.real_A: x_batch[:args.batch_size],
                                model.fake_A_sample: fake_A,
                                model.lr: curr_lr})
            
            # update D_B network
            d_B_loss, _ = sess.run([model.loss_d_B, model.d_B_trainer], 
                     feed_dict={model.real_B: x_batch[args.batch_size:],
                                model.fake_B_sample: fake_B,
                                model.lr: curr_lr})
                                        
            if step % 100 == 0:
                print('Epoch[{}/{}] Step[{}/{}] g_AtoB:{:.4f}, g_BtoA:{:.4f},\
                d_A:{:.4f}, d_B:{:.4f}'.format(epoch, args.epoch, step, steps_per_epoch, 
                                        g_AtoB_loss, g_BtoA_loss,
                                        d_A_loss, d_B_loss))

        # visualize
        fake_A, fake_B, global_step = sess.run([model.fake_A, model.fake_B, model.global_step],
                                 feed_dict={model.real_A: x_batch[:args.batch_size],
                                            model.real_B: x_batch[args.batch_size:]})
        
        
        filepath = os.path.join(model.log_dir, str(epoch) + '.png')
        visualize(x_batch, fake_A, fake_B, filepath)
                   
        # save model
        model.save(sess, args.model_dir, global_step)


def inference(args, model, sess):
    if args.model_dir is None:
        raise ValueError('Need to provide model directory')

    # load model
    model.load(sess, args.model_dir)
    
    # load data
    filenames = ld.load_filenames('./horse2zebra/test')
    
    batch_size = 16
    num_iter = min(len(filenames[0]), len(filenames[1])) // batch_size
    for i in range(num_iter):
        x_batch = ld.get_batch(filenames, batch_size, i, img['w'])  
        
        fake_A, fake_B = sess.run([model.fake_A, model.fake_B],
                                 feed_dict={model.real_A: x_batch[:batch_size],
                                            model.real_B: x_batch[batch_size:]})
        
        filepath = os.path.join(model.test_dir, str(i) + '.png')
        visualize(x_batch, fake_A, fake_B, filepath)


def cyclic_inference(args, model, sess):
    if args.model_dir is None:
        raise ValueError('Need to provide model directory')

    # load model
    model.load(sess, args.model_dir)
    
    # load data
    filenames = ld.load_filenames('./horse2zebra/test')
    
    batch_size = 16
    num_iter = min(len(filenames[0]), len(filenames[1])) // batch_size
    for i in range(num_iter):
        x_batch = ld.get_batch(filenames, batch_size, i, img['w'])  
        
        fake_A, fake_B_, fake_B, fake_A_ = sess.run([model.fake_A, model.fake_B_,
                                                     model.fake_B, model.fake_A_],
                                 feed_dict={model.real_A: x_batch[:args.batch_size],
                                            model.real_B: x_batch[args.batch_size:]})
        
        fake_B_A_ = np.concatenate([fake_B, fake_A_], axis=0)   # made from A
        fake_A_B_ = np.concatenate([fake_A, fake_B_], axis=0)   # made from B

        filepath = os.path.join(model.test_dir, str(i) + '.png')
        visualize(x_batch, fake_B_A_, fake_A_B_, filepath)

    

        