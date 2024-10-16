import torch.utils.data
from batch import BatchMasking

from util import MaskAtom
class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, mask_rate=0.0, **kwargs):
        self._transform = MaskAtom(num_atom_type = 35, mask_rate = mask_rate)
        super(DataLoaderMaskingPred, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)
    
    def collate_fn(self, batches):
        # batchs_origin = batches
        batchs = [self._transform(x) for x in batches]
        return BatchMasking.from_data_list(batchs)
  