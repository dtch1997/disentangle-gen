# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:28:06 2019

@author: Daniel
"""

import tensorflow as tf
import tensorflow.keras as tfk

class DisentanglementMetric(tfk.Model):
    
    def __init__(self, generator, classifier):
        """
        Args:
            - generator: A subclass of AbstractGenerator 
            - classifier: A subclass of AbstractClassifier
        """
        super().__init__()
        self.generator = generator
        self.classifier = classifier
            
    def forward(self, SI, ZI, ZnotI):
        """
        All batch sizes should be the same. 
        Return I(SI, ZnotI) - I(SI, ZI), which can be used as a loss. 
        """
        batch_size = SI.shape[0]
        ZnotI_sample, log_pZnotI = self.generator.sample_ZnotI(batch_size)
        X_given_ZI_samples = self.generator.generate(ZI, ZnotI_sample)
        p_SI_given_ZI = self.classifier.log_prob(X_given_ZI_samples, SI)
            # Tensor(batch_size)
        ZI_sample, log_pZI = self.generator.sample_ZI(batch_size)
        X_given_ZnotI_samples = self.generator.generate(ZI, ZnotI_sample)
        p_SI_given_ZnotI = self.classifier.log_prob(X_given_ZnotI_samples, SI)
            # Tensor(batch_size)
        p_SI = None # TODO
        
        I_SI_ZI = tf.math.reduce_sum(tf.math.log(tf.math.divide(p_SI_given_ZI, p_SI)))
        I_SI_ZnotI = tf.math.reduce_sum(tf.math.log(tf.math.divide(p_SI_given_ZnotI, p_SI)))
        return I_SI_ZnotI - I_SI_ZI
        
            
            