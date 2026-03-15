# from .jasper_ridge import JasperRidgeDataset, MotionCodeJasperRidge
# from .urban import UrbanDataset, MotionCodeUrban
from .jasper_ridge import JasperRidgeDataset
from .urban import UrbanDataset
from .grss import GRSSDataset

dataset_factory = {
    'jasper_ridge': JasperRidgeDataset,
    # 'jasper_ridge_pixel': MotionCodeJasperRidge,
    'urban': UrbanDataset,
    # 'urban_pixel': MotionCodeUrban
    'grss': GRSSDataset,
}
