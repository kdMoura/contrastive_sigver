import os
from sigver.datasets.base import IterableDataset
from skimage.io import imread
from skimage import img_as_ubyte


class CedarDataset(IterableDataset):
    """ Helper class to load the CEDAR dataset
    """
    def __init__(self, path):
        self.path = path
        self.users = list(range(1, 55+1))

    @property
    def genuine_per_user(self):
        return 24

    @property
    def skilled_per_user(self):
        return 24

    @property
    def simple_per_user(self):
        return 0

    @property
    def maxsize(self):
        return 730, 1042

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        files = ['{}_{}_{}.png'.format('original', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_org', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        files = ['{}_{}_{}.png'.format('forgeries', user, img) for img in range(1, 24 + 1)]
        for f in files:
            full_path = os.path.join(self.path, 'full_forg', f)
            img = imread(full_path, as_gray=True)
            yield img_as_ubyte(img), f

    def iter_simple_forgery(self, user):
        yield from ()  # No simple forgeries
    
    def get_signature(self, user, img_idx, forgery):
        """
        Returns a particular signature given by:
        - user id
        - image index
        - forgery flag
        """
    
        if forgery:
            folder = 'full_forg'
            filename = f'forgeries_{user}_{img_idx}.png'
        else:
            folder = 'full_org'
            filename = f'original_{user}_{img_idx}.png'
    
        full_path = os.path.join(self.path, folder, filename)

        return img_as_ubyte(imread(full_path, as_gray=True))
