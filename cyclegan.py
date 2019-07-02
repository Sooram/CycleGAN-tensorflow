# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:47:33 2019

@author: Sooram Kang

"""
import os
import tensorflow as tf
from general_model import Model

img = {
       'w': 256,
       'h': 256,
       'c': 3
       }

class CycleGAN(Model):
    def __init__(self, args):

        #########################
        #                       #
        #    General Setting    #
        #                       #
        #########################

        self.args = args

        self.model_dir = args.model_dir

        if not self.model_dir:
            raise ValueError('Need to provide model directory')

        self.log_dir = os.path.join(self.model_dir, 'log')
        self.test_dir = os.path.join(self.model_dir, 'test')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.global_step = tf.train.get_or_create_global_step()
        
        #########################
        #                       #
        #     Model Building    #
        #                       #
        #########################
        
        self.L1_lambda = 10.0
        
        if(img['w'] == 128):
            self.n_res = 6
        else:
            self.n_res = 9
             
        net_G = Generator()
        net_D = Discriminator()

        # 1. Build Generator
        
        self.real_A = tf.placeholder(tf.float32, shape=[None, img['w'], img['h'], img['c']])
        self.real_B = tf.placeholder(tf.float32, shape=[None, img['w'], img['h'], img['c']])

        self.fake_B = net_G(self.real_A, self.n_res, name="g_AtoB")    # A->B
        self.fake_A_ = net_G(self.fake_B, self.n_res, name="g_BtoA")   # A->B->A_
        self.fake_A  = net_G(self.real_B, self.n_res, name="g_BtoA")   # B->A
        self.fake_B_  = net_G(self.fake_A, self.n_res, name="g_AtoB")  # B->A->B_

        
        # 2. Build Discriminator
        
        d_real_A = net_D(self.real_A, name="d_A")   
        d_real_B = net_D(self.real_B, name="d_B")   
        d_fake_A = net_D(self.fake_A, name="d_A")   
        d_fake_B = net_D(self.fake_B, name="d_B")
        
        
        # 3. Calculate loss        
        
        # 3-1. Generator loss
        # adversarial loss
        loss_g_adv_A = tf.reduce_mean(tf.squared_difference(d_fake_A, tf.ones_like(d_fake_A)))
        loss_g_adv_B = tf.reduce_mean(tf.squared_difference(d_fake_B, tf.ones_like(d_fake_B)))
        
        # cyclic loss
        loss_cyc_A = tf.reduce_mean(tf.abs(self.real_A - self.fake_A_))
        loss_cyc_B = tf.reduce_mean(tf.abs(self.real_B - self.fake_B_))
        self.loss_cyc = loss_cyc_A + loss_cyc_B
        
        self.loss_g_AtoB = loss_g_adv_B + self.L1_lambda * self.loss_cyc
        self.loss_g_BtoA = loss_g_adv_A + self.L1_lambda * self.loss_cyc
        
        # 3-2. Descriminator loss
        self.fake_A_sample = tf.placeholder(tf.float32, [None, img['w'], img['h'], img['c']], 
                                            name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32, [None, img['w'], img['h'], img['c']], 
                                            name='fake_B_sample')
        
        d_fake_A_sample = net_D(self.fake_A_sample, name="d_A")
        d_fake_B_sample = net_D(self.fake_B_sample, name="d_B")

        loss_d_real_A = tf.reduce_mean(tf.squared_difference(d_real_A, tf.ones_like(d_real_A))) 
        loss_d_fake_A = tf.reduce_mean(tf.squared_difference(d_fake_A_sample, tf.zeros_like(d_fake_A_sample)))
        self.loss_d_A= (loss_d_real_A + loss_d_fake_A) / 2     
        
        loss_d_real_B = tf.reduce_mean(tf.squared_difference(d_real_B, tf.ones_like(d_real_B)))
        loss_d_fake_B = tf.reduce_mean(tf.squared_difference(d_fake_B_sample, tf.zeros_like(d_fake_B_sample)))
        self.loss_d_B = (loss_d_real_B + loss_d_fake_B) / 2     
        
        # 4. Update weights
        
        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_AtoB' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_BtoA' in var.name]

        optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.99)

        self.d_A_trainer = optim.minimize(self.loss_d_A, var_list=d_A_vars)
        self.d_B_trainer = optim.minimize(self.loss_d_B, var_list=d_B_vars)
        self.g_A_trainer = optim.minimize(self.loss_g_AtoB, var_list=g_A_vars)
        self.g_B_trainer = optim.minimize(self.loss_g_BtoA, var_list=g_B_vars)
            
        # 5. Visualize
#        tf.summary.image('RealA', self.real_A)
#        tf.summary.image('FakeB', self.fake_B)
#        tf.summary.image('CycA', self.fake_A_)
#        tf.summary.image('RealB', self.real_B)
#        tf.summary.image('FakeA', self.fake_A)
#        tf.summary.image('CycB', self.fake_B_)

#        with tf.name_scope('Generator'):
#            tf.summary.scalar('g_BtoA_loss', self.loss_g_BtoA)
#            tf.summary.scalar('g_AtoB_loss', self.loss_g_AtoB)
#            tf.summary.scalar('cyc_loss', self.loss_cyc)
#        with tf.name_scope('Discriminator'):
#            tf.summary.scalar('d_total_loss', self.loss_d)
#            tf.summary.scalar('d_A_loss', self.loss_d_A)
#            tf.summary.scalar('d_B_loss', self.loss_d_B)
#
#        self.summary_op = tf.summary.merge_all()
            
        super(CycleGAN, self).__init__(5)

#%%
def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset
  
class Discriminator(object):        
    def __call__(self, inputs, name="discriminator"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            d1 = tf.layers.conv2d(inputs, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='SAME')
            d1 = self.lrelu(d1)
            d2 = tf.layers.conv2d(d1, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='SAME')
            d2 = self.lrelu(instance_norm(d2, name="d2"))
            d3 = tf.layers.conv2d(d2, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='SAME')
            d3 = self.lrelu(instance_norm(d3, name="d3"))
            d4 = tf.layers.conv2d(d3, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='SAME')
            d4 = self.lrelu(instance_norm(d4, name="d4"))
            d5 = tf.layers.conv2d(d4, filters=1, kernel_size=[4, 4], strides=(1, 1), padding='SAME')
#            d5 = tf.nn.sigmoid(d5)  # not in paper
            
            return d5
            
    def lrelu(self, x, leak=0.2, name="lrelu"):
        return tf.maximum(x, leak*x)
            
"""
The network with 6 residual blocks consists of:
c7s1-64,d128,d256,R256,R256,R256,
R256,R256,R256,u128,u64,c7s1-3
The network with 9 residual blocks consists of:
c7s1-64,d128,d256,R256,R256,R256,
R256,R256,R256,R256,R256,R256,u128
u64,c7s1-3            
"""
class Generator(object):
    def __call__(self, inputs, n_res, name="generator"):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            """ encoder """
            e0 = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            e1 = tf.layers.conv2d(e0, filters=64, kernel_size=[7, 7], strides=(1, 1), padding='VALID')
            e1 = tf.nn.relu(instance_norm(e1, name="e1"))
            e2 = tf.layers.conv2d(e1, filters=128, kernel_size=[3, 3], strides=(2, 2), padding='SAME')
            e2 = tf.nn.relu(instance_norm(e2, name="e2"))
            e3 = tf.layers.conv2d(e2, filters=256, kernel_size=[3, 3], strides=(2, 2), padding='SAME')
            e3 = tf.nn.relu(instance_norm(e3, name="e3"))
            
            """ transformer """
            r = e3
            for i in range(n_res):
                r = self.residual_block(r)


            """ decoder """
            d1 = tf.layers.conv2d_transpose(r, 128, [3,3], strides=(2, 2), padding='SAME')
            d1 = tf.nn.relu(instance_norm(d1, name="g_d1"))
            d2 = tf.layers.conv2d_transpose(d1, 64, [3,3], strides=(2, 2), padding='SAME')
            d2 = tf.nn.relu(instance_norm(d2, name="g_d2"))
            # reflect or same?
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            d3 = tf.layers.conv2d(d2, filters=3, kernel_size=[7, 7], strides=(1, 1), padding='VALID')
#            d3 = tf.nn.relu(instance_norm(d3))  # paper
            d3 = tf.nn.tanh(d3)                 # codes

            return d3
            
    def residual_block(self, inputs):
        r0 = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        r1 = tf.layers.conv2d(r0, filters=256, kernel_size=[3, 3], strides=(1, 1), padding='VALID')
        r1 = tf.nn.relu(instance_norm(r1, name="r1"))
        
        r2 = tf.pad(r1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
        r3 = tf.layers.conv2d(r2, filters=256, kernel_size=[3, 3], strides=(1, 1), padding='VALID')
        r3 = instance_norm(r3, name="r3")
        
        return r3 + inputs
    
