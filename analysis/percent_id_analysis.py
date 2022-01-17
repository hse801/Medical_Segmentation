import os
import nibabel as nb
import numpy as np
import tables
import SimpleITK as sitk
import glob
import torch
import pydicom
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interpn
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pylab as pylab

"""

plot %ID correlation
compare unet, attentionunet, proposed method
R2 coefficient, pearson coefficient

"""

percent_ids = pd.read_csv('./%ID_analysis_210_new.csv')
# print(percent_ids)

# Data filtering(less outlier)
# percent_ids = percent_ids[0.03 > percent_ids['mask']]

# pearson correlation coefficient
unet_pearson = percent_ids['mask'].corr(percent_ids['unet'], method='pearson')
attention_pearson = percent_ids['mask'].corr(percent_ids['attention'], method='pearson')
proposed_pearson = percent_ids['mask'].corr(percent_ids['proposed'], method='pearson')

# R squared
unet_r2 = r2_score(percent_ids['mask'], percent_ids['unet'])
attention_r2 = r2_score(percent_ids['mask'], percent_ids['attention'])
proposed_r2 = r2_score(percent_ids['mask'], percent_ids['proposed'])

# percent_ids = percent_ids[0.015 >percent_ids['mask']]
# percent_ids = percent_ids[0.01 >percent_ids['proposed']]

sns.regplot(x='mask', y='proposed', ci=None, color='blue', marker='o', data=percent_ids)
sns.scatterplot(x='mask', y='proposed', color='black', data=percent_ids)

plt.xlabel('Mask (%ID)', fontsize=15)
plt.ylabel('Proposed (%ID)', fontsize=15)
z = np.polyfit(percent_ids['mask'], percent_ids['proposed'], 1)
plt.legend([f'y = {z[0]:.4f}x + {z[1]:.4f} [R2 = {proposed_r2:.4f}]'], fontsize=13.5)

plt.savefig('./0117_r2_proposed.jpg')

# sns.scatterplot(x='mask', y='attention', color='blue', data=percent_ids)
# sns.regplot(x='mask', y='attention', ci=None, data=percent_ids)
# percent_ids.plot(kind='scatter', x='mask', y='proposed')
# fit_weight = np.polyfit(percent_ids['mask'], percent_ids['proposed'], 1)
# trend_f = np.poly1d(fit_weight)

# plt.plot(percent_ids['mask'], trend_f(percent_ids['proposed']),"g.")


# plt.xlim(0, 0.1)
# plt.ylim(0, 0.1)
# # plt.show()

# Save plot for UNET
# z = np.polyfit(percent_ids['mask'], percent_ids['unet'], 1)
# p = np.poly1d(z)
# pylab.plot(percent_ids['mask'], percent_ids['unet'], 'o')
# pylab.plot(percent_ids['mask'], p(percent_ids['mask']), 'r--')
# plt.title("y={:.6f}x+({:.6f}) [R2={:.6f}]".format(z[0], z[1], unet_r2))
# plt.xlabel('Mask')
# plt.ylabel('U-Net')
# print( f"y = {z[0]}x + {z[1]}")
# plt.savefig('./unet_r2_210_new.jpg')

# Save plot for ATTENTIONUNET
# z = np.polyfit(percent_ids['mask'], percent_ids['attention'], 1)
# p = np.poly1d(z)
#
# pylab.plot(percent_ids['mask'], percent_ids['attention'], 'o')
# pylab.plot(percent_ids['mask'], p(percent_ids['mask']), 'r--')
# plt.title("y={:.6f}x+({:.6f}) [R2={:.6f}]".format(z[0], z[1], attention_r2))
# plt.xlabel('Mask')
# plt.ylabel('Attention U-Net')
# print( f"y = {z[0]}x + {z[1]}")
# plt.savefig('./attention_r2_210_new.jpg')

# Save plot for PROPOSED
# z = np.polyfit(percent_ids['mask'], percent_ids['proposed'], 1)
# p = np.poly1d(z)
#
# pylab.plot(percent_ids['mask'], percent_ids['proposed'], 'o')
# pylab.plot(percent_ids['mask'], p(percent_ids['mask']), 'r--')
# plt.title("y={:.6f}x+({:.6f}) [R2={:.6f}]".format(z[0], z[1], proposed_r2))
# plt.xlabel('Mask')
# plt.ylabel('Proposed')
# print( f"y = {z[0]}x + {z[1]}")
# plt.savefig('./proposed_r2_210_new.jpg')

print(f'unet_pearson = {unet_pearson:.5f}, attention_pearson = {attention_pearson:.5f}, proposed_pearson = {proposed_pearson:.5f}')
print(f'unet_r2 = {unet_r2:.5f}, attention_r2 = {attention_r2:.5f}, proposed_r2 = {proposed_r2:.5f}')


