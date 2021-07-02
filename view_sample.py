import numpy as np
import matplotlib.pyplot as plt

mri = np.load('sample/val/case02/MR.npy')
print(mri.shape)
plt.imshow(mri[6,:,:])
plt.show()