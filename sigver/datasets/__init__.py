from .gpds import GPDSDataset
from .gpds_synth import GPDSSynthDataset
from .mcyt import MCYTDataset
from .cedar import CedarDataset
from .brazilian import BrazilianDataset, BrazilianDatasetWithoutSimpleForgeries
from .bhsig260_bengali import BengaliDataset
from .bhsig260_hindi import HindiDataset

available_datasets = {'gpds': GPDSDataset,
                      'gpds_synth': GPDSSynthDataset,
                      'mcyt': MCYTDataset,
                      'cedar': CedarDataset,
                      'brazilian': BrazilianDataset,
                      'brazilian-nosimple': BrazilianDatasetWithoutSimpleForgeries,
                      'bengali':BengaliDataset,
                      'hindi':HindiDataset}