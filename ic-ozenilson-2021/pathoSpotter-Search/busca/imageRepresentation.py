import json


class ImageRepresentation(object):
    
    def __init__(self, name, image_path):
        self.similarity = 0
        self.representations = dict()
        self.name = name
        self.path = image_path
        
    def add_representation(self, name_representation, representation):
        self.representations[name_representation] = representation
        
    def get_representation(self, name_representation):
        return self.representations[name_representation]
    
    def to_dict(self):
        aux = {'path':self.path, 'name':self.name, self.name:self.representations}
        return aux
    
        
    