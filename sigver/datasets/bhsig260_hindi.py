#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from skimage.io import imread
from sigver.datasets.base import IterableDataset
from skimage import img_as_ubyte


class HindiDataset(IterableDataset):
    """ Helper class to load the BHSig260 Hindi
    """

    def __init__(self, path, extension='tif'):
        self.path = path
        self.users = [int(user) for user in sorted(os.listdir(self.path)) if user.isdigit()]
        self.extension = extension

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 30

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 936, 1329
        #return 435 1329
        #1329/1.42 = 936

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""
        
        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_genuine_files = filter(lambda x: '-G-' in x, all_files)
        for f in user_genuine_files:
            full_path = os.path.join(user_folder, f)
            #print('DEBUG: ',full_path)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_folder = os.path.join(self.path, '{:03d}'.format(user))
        all_files = sorted(os.listdir(user_folder))
        user_forgery_files = filter(lambda x: '-F-' in x, all_files)
        for f in user_forgery_files:
            full_path = os.path.join(user_folder, f)
            #print('DEBUG: ',full_path)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def get_signature(self, user, img_idx, forgery):
        """ Returns a particular signature (given by user id, img id and
            whether or not it is a forgery
        """
        if forgery:
            c = 'F'
        else:
            c = 'G'
        filename = 'B-S-{}-{}-{:02d}.{}'.format(user,c, img_idx,
                                                self.extension)
        full_path = os.path.join(self.path, '{:03d}'.format(user), filename)
        return img_as_ubyte(imread(full_path, as_gray=True))

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries

