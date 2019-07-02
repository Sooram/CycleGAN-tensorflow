# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:20:18 2019

@author: Sooram Kang

"""

import argparse
import tensorflow as tf
from cyclegan import CycleGAN
from train_test import train, inference

parser = argparse.ArgumentParser(description='CycleGAN')
parser.add_argument('--model_dir', type=str, 
                      default='./exp1',
                      help='Directory in which the model is stored')
parser.add_argument('--data_dir', type=str,
                      default='./horse2zebra/train',
                      help='Directory in which the data is stored')
parser.add_argument('--is_training', type=bool, default=True, help='whether it is training or inferecing')
parser.add_argument('--batch_size', type=int, default=1, help='batch size per domain')
parser.add_argument('--epoch', type=int, default=200, help='epochs')
parser.add_argument('--start_epoch', type=int, default=93, help='epochs')
parser.add_argument('--max_size', type=int, default=50, help='number of images needed to update the descriminator')
parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate for discriminator')
#parser.add_argument('--g_lr', type=float, default=1e-3, help='learning rate for generator')


        
#%%
def main(args):
    # build model 
    model = CycleGAN(args)
    
    # open session 
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = "0"
    
    sess = tf.Session(config=c)
    sess.run(tf.global_variables_initializer())

    train(args, model, sess) if args.is_training else inference(args, model, sess)


    
if __name__ == '__main__':
    config, unparsed = parser.parse_known_args()
    main(config)