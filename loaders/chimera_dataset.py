from torch.utils.data import Dataset


class Chimera(Dataset):
    def __init__(self, ds, mode='horizontal'):
        self.ds = ds
        self.mode = mode

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, lbl = self.ds[i]
        if self.mode == 'horizontal':
            half = (img.shape[2]) // 2
            img2, lbl2 = self.select_diff_digit(lbl, i)

            img[:, :half, :] = img2[:, :half, :]

            img[:, half-3:half+3, :] = 0

            return img, str(f'{lbl},{lbl2}')

        elif self.mode == 'horizontal_blank':
            half = (img.shape[2]) // 2
            if i % 2 == 0:
                img[:, :half, :] = 0.
            else:
                img[:, half:, :] = 0.
            return img, lbl
        else:
            raise ValueError

    def select_diff_digit(self, digit, i):
        new_digit = digit
        idx = i
        while new_digit == digit:
            idx += 1
            idx = idx % len(self.ds)
#             idx = np.random.randint(len(self.ds))
            img , lbl = self.ds[idx]
            new_digit = lbl
        return img, lbl
