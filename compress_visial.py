import torch
import numpy as np
from matplotlib import pyplot as plt
from simulation.train_code.utils import LoadTraining_npy_spec_polo


# filter_path = 'self_made_matrix.npy' #15 21 4 200 200
filter_path = 'mask/tio2_100_mutual_10_10.npy' #15 21 4 200 200
filter_matrix_test = torch.from_numpy(np.load(filter_path)).unsqueeze(0).repeat(15, 1, 1, 4, 4).cuda().float()
filter_matrix_test = filter_matrix_test.cpu().numpy()
test_data = LoadTraining_npy_spec_polo('F:\denoise_test')
test_data = np.stack(test_data, axis=0).transpose(0, 3, 4, 1, 2)  # 15 21 4 512 612

bs, c, s, h, w = test_data.shape
filter_matrix_test = filter_matrix_test[:, :, :, :h, :w]

compress = test_data * filter_matrix_test
compress = np.sum(compress, axis=(1, 2))

data = compress[6, :, :]
plt.figure()
plt.imshow(data, cmap='gray')
plt.show()


# for index in range(15):
#     data = compress[index, :, :]
#     plt.figure()
#     plt.imshow(data, cmap='gray')
#     plt.show()