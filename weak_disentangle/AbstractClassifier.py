# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:52:15 2019

@author: Daniel
"""

class AbstractClassifier:
    """
    A thin wrapper over a generic classifier. 
    Subclass this and implement the methods to interface with DisentanglementMetric
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def log_prob(self, X, SI):
        """
        Returns the log probability of SI given X
        """
        raise NotImplementedError