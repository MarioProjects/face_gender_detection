import os
import cv2
import pandas as pd
import json
import numpy as np

import torch
from torch.utils import data

import pytorchlib.pytorch_data.load_data as load_data
import pytorchlib.pytorch_library.utils_training as utils_training

import albumentations

import PIL

IMG_BASE_SIZE = 100
CAT2CLASS = {0:"Male", 1:"Female"}

class NPDatasetLFW(data.Dataset):
    """
        Cargador de prueba de dataset almacenado en carpetas del modo
            -> train/clase1/TodasImagenes ...
        data_path: ruta a la carpeta padre del dataset. Ejemplo train/
        transforms: lista de transformaciones de albumentations a aplicar
        cat2class: diccionario con claves clas clases y valor la codificacion
            de cada clase. Ejemplo {'perro':0, 'gato':1}
    """
    def __init__(self, data_path, gray_images=False, exclude_fold=[0], transforms=[], normalization="", seed=0):

        # '/home/maparla/pytorch_datasets/nplfwdeepfunneled_gray/'
        '''
        LOAD LFW DATASET
        Carga los 5 folds de LFW y y de ellos toma 1 ("fold_test") como test
        '''

        """ Extraemos los datos """
        features, labels = [], []

        for i in range(0,5): # Existen 5 folds en LFW
            if i not in exclude_fold:
                features.append(np.load(data_path+"fold"+str(i)+"_data.npy"))
                labels.append(np.load(data_path+"fold"+str(i)+"_labels.npy"))

        features = np.concatenate(features)
        labels = np.concatenate(labels)

        # Debemos hace el transpose para poner los canales delante
        features = features.astype(np.float32).transpose(0,3,1,2)
        labels = labels.astype('int64').reshape(-1, 1)

        """ Ordenamos los datos según la semilla, los barajamos (Train) """
        generator = np.random.RandomState(seed=seed)
        index = generator.permutation(features.shape[0])
        features = features[index]
        labels = labels[index]

        # Los pasamos a pytorch los datos
        features = torch.from_numpy(features).cuda()
        labels = torch.from_numpy(labels).cuda()

        # Normalizamos los datos y los añadimos al Dataset
        self.transforms = transforms
        self.normalization = normalization
        self.features = load_data.single_normalize(features, self.normalization)
        self.labels = labels

    def __getitem__(self,index):

        feature = self.features[index]

        if self.transforms!=[]:
            # para aplicar las transformaciones necesitamos usar numpy
            feature = feature.data.cpu().numpy().transpose(1,2,0)
            for transform in self.transforms:
                feature = load_data.apply_img_albumentation(transform, feature)
            # Volvemos a pasar a pytorch tensor poniendo el canal delante
            feature = torch.from_numpy(feature.astype(np.float32).transpose(2,0,1))

        return feature, self.labels[index]

    def __len__(self):
        return len(self.features)