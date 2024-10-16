import torch
import random
import numpy as np

class MaskAtom:
    def __init__(self, num_atom_type, mask_rate):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.mask_rate = mask_rate

    def feature_masking(self, size):
        one_hot_vector = np.zeros(size, dtype=int)
        one_hot_vector[-1] = 1
        return one_hot_vector.tolist()
    
    def __call__(self, data, masked_atom_indices=None):
        
        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.labels_atoms[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor(self.feature_masking(36) + # Mask atom type
                                             self.feature_masking(9) + # Mask degree
                                                self.feature_masking(7) +  # Mask Orbital hybridization
                                                    self.feature_masking(6)) # Mask Number of hydrogens

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, mask_rate={})'.format(
            self.__class__.__name__, self.num_atom_type,
            self.mask_rate)

