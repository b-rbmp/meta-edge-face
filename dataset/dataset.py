from collections import defaultdict
import numbers
import os

import numpy as np
np.bool = np.bool_
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, filename, transform, min_samples_per_identity=1):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, f'{filename}.rec')
        path_imgidx = os.path.join(root_dir, f'{filename}.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        
        # Read the first record to see how the indices are stored
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            # If flag > 0, we can extract the number of samples from header
            # header.label might contain [num_samples, ...] in some datasets
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            # Otherwise, just take all keys from the MXIndexedRecordIO object
            self.imgidx = np.array(list(self.imgrec.keys))
        
        # ---------------------------------------------------------------------
        # Count number of samples per identity
        # ---------------------------------------------------------------------
        label_counts = defaultdict(int)
        
        # First pass: count how many samples each identity has
        for idx in self.imgidx:
            s = self.imgrec.read_idx(idx)
            header, _ = mx.recordio.unpack(s)
            label = header.label
            # If the label is an array, take the first element
            if not isinstance(label, numbers.Number):
                label = label[0]
            label_counts[label] += 1
        
        # Collect valid identities that meet the threshold
        valid_identities = {lbl for lbl, count in label_counts.items()
                            if count >= min_samples_per_identity}
        
        # Filter out image indices that do not belong to valid identities
        filtered_indices = []
        for idx in self.imgidx:
            s = self.imgrec.read_idx(idx)
            header, _ = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            if label in valid_identities:
                filtered_indices.append(idx)
        
        # Overwrite the original indices with the filtered version
        self.imgidx = np.array(filtered_indices)
        
        # Calculate and print how many identities got removed
        total_identities = len(label_counts)
        valid_count = len(valid_identities)
        num_identities_skipped = total_identities - valid_count
        if total_identities > 0:
            percentage_skipped = 100.0 * num_identities_skipped / total_identities
        else:
            percentage_skipped = 0
        
        print(
            f"Removed {num_identities_skipped} identities "
            f"({percentage_skipped:.2f}% of total) "
            f"because they had fewer than {min_samples_per_identity} samples."
        )

        bookkeeping_path = os.path.join(self.root_dir, f"bookkeeping_{min_samples_per_identity}.pkl")
        if os.path.exists(bookkeeping_path):
            print(f"Bookkeeping file found at {bookkeeping_path}")
        else:
            print(f"Bookkeeping file not found at {bookkeeping_path}. Will be generated and cached")

        self._bookkeeping_path = bookkeeping_path


    def __getitem__(self, index):
        """
        Return the sample (image) and label for the item at `index`.
        """
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        
        # Decode the image
        sample = mx.image.imdecode(img).asnumpy()
        
        # Apply transforms if given
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label

    def __len__(self):
        return len(self.imgidx)