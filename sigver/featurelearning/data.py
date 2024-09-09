import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as transforms

from sigver.featurelearning.models.signet import SigNet
from sigver.datasets.util import load_dataset


class TransformDataset(Dataset):
    """
        Dataset that applies a transform on the data points on __get__item.
    """
    def __init__(self, dataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index
        
        try:
            self.targets = dataset[:][1]
        except IndexError:
            self.targets = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]

        return tuple((self.transform(img), *data[1:]))


def extract_features(x, process_function, batch_size, input_size=None):
    data = TensorDataset(torch.from_numpy(x))

    if input_size is not None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

        data = TransformDataset(data, data_transforms)

    data_loader = DataLoader(data, batch_size=batch_size)
    result = []

    with torch.no_grad():
        for batch in data_loader:
            result.append(process_function(batch))
    return torch.cat(result).cpu().numpy()

def get_features(model_path, data, gpu_idx=0, input_size=(150, 220), batch_size=32):
    
    state_dict, class_weights, forg_weights = torch.load(model_path,map_location=lambda storage, loc: storage)
    device = torch.device('cuda', gpu_idx) if torch.cuda.is_available() else torch.device('cpu')

    print('Using device: {}'.format(device))

    #base_model = models.available_models['signet']().to(device).eval()
    base_model = SigNet().to(device).eval()

    base_model.load_state_dict(state_dict)

    def process_fn(batch):
        input = batch[0].to(device)
        return base_model(input)

    if isinstance(data, str):
        # if data is a path, load preprocessed data
        x, y, yforg, user_mapping, filenames = load_dataset(data)
    
        features = extract_features(x, process_fn, batch_size, input_size)

        return features, y, yforg, user_mapping, filenames
    
    # if not path, extract features from provided data
    return extract_features(data, process_fn, batch_size, input_size)
    