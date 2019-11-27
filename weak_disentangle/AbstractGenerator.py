# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:33:45 2019

@author: Daniel
"""

class AbstractGenerator:
    """
    A thin wrapper over a generic generator. 
    Subclass this and implement the methods to interface with DisentanglementMetric
    """

    def __init__(self):
        raise NotImplementedError
        
    def generate(self, ZI, ZnotI):
        """
        Args: A batch of latent variables
        Return: A batch of samples based on ZI, ZnotI
        """
        raise NotImplementedError
        
    def sample_ZI(self, batch_size: int):
        """
        Samples from the prior of ZI
        
        Args: 
            - batch_size
            
        Return:
            - tf.Tensor() of shape (batch_size, ZI_dim), log p(ZI)
        """
        raise NotImplementedError
        ZI, log_prob = None, None
        return ZI, log_prob
        
    def sample_ZnotI(self):
        """
        Samples from the prior of ZnotI
        
        Args: 
            - batch_size
            
        Return:
            - tf.Tensor() of shape (batch_size, ZnotI_dim)
        """
        raise NotImplementedError
        ZnotI, log_prob = None, None
        return ZnotI, log_prob
        
        