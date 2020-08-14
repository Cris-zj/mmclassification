from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import CrossEntropyLoss, cross_entropy
from .label_smooth_loss import LabelSmoothLoss, label_smooth
from .mse_loss import MSELoss, mse_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .soft_cross_entropy_loss import SoftCrossEntropyLoss, soft_cross_entropy
from .soft_label_smooth_loss import SoftLabelSmoothLoss, soft_label_smooth

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'label_smooth', 'LabelSmoothLoss', 'weighted_loss',
    'MSELoss', 'mse_loss', 'SoftCrossEntropyLoss', 'soft_cross_entropy',
    'SoftLabelSmoothLoss', 'soft_label_smooth'
]
