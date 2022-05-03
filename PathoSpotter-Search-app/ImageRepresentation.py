#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:34:08 2019

@author: gabriel
Classe que implmenta a representacao de uma imagem
"""

import json


class ImageRepresentation(object):

    def __init__(self, name, image_path):
        self.representations = dict()
        self.name = name
        self.path = image_path

    #    def cmp(self, x, y):
    #        """
    #        Replacement for built-in function cmp that was removed in Python 3
    #
    #        Compare the two objects x and y and return an integer according to
    #        the outcome. The return value is negative if x < y, zero if x == y
    #        and strictly positive if x > y.
    #        """
    #        return (x > y) - (x < y)

    def add_representation(self, name_representation, representation):
        self.representations[name_representation] = representation

    def get_representation(self, name_representation):
        return self.representations[name_representation]

    #    def __cmp__(self, other):
    #        '''
    #        Metodo usado para manter a priority queue ordenado
    #        '''
    #        return self.cmp(self.simalarity, other.similarity)

    def text_information(self):
        return ('Nome da imagem: %s; distância: %f' % (self.name, self.similarity))

    def to_json(self):
        aux = {'path': self.path, 'name': self.name, 'similarity': self.similarity}
        return json.dumps(aux)
