# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:52:15 2019

@author: Daniel
"""

class AbstractClassifier:
    
    def __init__(self):
        raise NotImplementedError
        
    def log_prob(self, X, SI):
        """
        Returns the log probability of SI given X
        """
        raise NotImplementedError