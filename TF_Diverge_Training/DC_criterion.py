import tensorflow as tf

import numpy as np
from contextlib import nullcontext

class Loss_DC(tf.keras.losses.Loss):
    def __init__(self,alpha=0.1):
        super(Loss_DC,self).__init__()
        self.ce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.alpha = alpha
        print("Loss balance alpha is: ", alpha)
    def CE(self,y_true,y_pred):
        return self.ce(y_true,y_pred) #the reversed order of that in Pytorch!!!

    def Distance_Correlation(self, latent, control):
        matrix_a = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.expand_dims(latent,axis=0)  -  tf.expand_dims(latent,1)),axis=-1) + 1e-12)
        matrix_b = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.expand_dims(control,axis=0) -  tf.expand_dims(control,1)),axis=-1) + 1e-12)
        # print((tf.math.reduce_sum(tf.math.square(tf.expand_dims(latent,axis=0)  -  tf.expand_dims(latent,1)),axis=-1)).shape)
        matrix_A = matrix_a - tf.math.reduce_mean(matrix_a, axis = 0, keepdims= True) - tf.math.reduce_mean(matrix_a, axis = 1, keepdims= True) + tf.math.reduce_mean(matrix_a)
        matrix_B = matrix_b - tf.math.reduce_mean(matrix_b, axis = 0, keepdims= True) - tf.math.reduce_mean(matrix_b, axis = 1, keepdims= True) + tf.math.reduce_mean(matrix_b)
        # print(tf.math.reduce_mean(matrix_a, axis = 0, keepdims= True).shape)
        Gamma_XY = tf.math.reduce_sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_XX = tf.math.reduce_sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
        Gamma_YY = tf.math.reduce_sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])

        correlation_r = Gamma_XY/tf.math.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r


    def __call__(self,y_pred,y_true,latent,controls):
        cls_loss = self.ce(y_true,y_pred)
        dc_loss = 0
        DC_results = []
        for control in controls:
            DC = self.Distance_Correlation(latent,control)
            dc_loss += DC
            DC_results.append(DC.numpy())
        if len(controls)==0:
            dc_loss =0
        else:
            dc_loss /= len(controls)+1e-12 
        loss = cls_loss + self.alpha*dc_loss
        
        return loss,cls_loss,dc_loss,DC_results


def run_nets(nets, idx, inputs, targets, criterion):
    with tf.GradientTape() as tape:
        eval_sub_net = list(range(idx))
        ref_features = []
        for sub_net_idx in eval_sub_net:
            _, feature = nets[sub_net_idx](inputs)
            ref_features.append(feature.numpy())

        outputs, learned_feature = nets[idx](inputs,training=True) 
        loss, _, _, DC_results = criterion(outputs, targets, learned_feature, ref_features)
        grads = tape.gradient(loss,nets[idx].trainable_weights) 
             
        if len(DC_results) < len(nets):
            for _ in range(len(nets) - 1 - len(DC_results)):
                DC_results.append(0.0)
        DC_results = np.asarray(DC_results)
        return outputs, loss, DC_results,grads


def eval_nets(nets, idx, inputs, targets, criterion):
    eval_sub_net = list(range(idx))
    ref_features = []
    for sub_net_idx in eval_sub_net:
        _, feature = nets[sub_net_idx](inputs)
        ref_features.append(feature.numpy())

    outputs, learned_feature = nets[idx].predict(inputs,verbose=0)
    loss, _, _, DC_results = criterion(outputs, targets, learned_feature, ref_features)
        
    if len(DC_results) < len(nets):
        for _ in range(len(nets) - 1 - len(DC_results)):
            DC_results.append(0.0)
    DC_results = np.asarray(DC_results)
    return outputs, loss, DC_results



