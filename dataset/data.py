# exibir as imagens negative e positve da pasta dataset 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# pasta onde estão as imagens
path = 'SteelDefect/data'


# lista de imagens
images = os.listdir(path)

plt.figsize=(15, 10)
# exibir apenas 1x10 imagens e salvar
fig, ax = plt.subplots(1, 10, figsize=(15, 10))
for j in range(10):
    image = cv2.imread(os.path.join(path, images[j+7]))
    h, w, _ = image.shape
    if h > w:
        image = image[:w, :]
    else:
        image = image[:, :h]
    ax[j].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[j].axis('off')


plt.tight_layout()
plt.show()
fig.savefig('dataset.png', bbox_inches = 'tight', pad_inches = 0)





# Numero de imagens positivas e negativas

# pasta onde estão as imagens
# path1 = 'dataset/Positive'
# path2 = 'dataset/Negative'

# lista de imagens
# images1 = os.listdir(path1)
# images2 = os.listdir(path2)

# print('Positive:', len(images1))
# print('Negative:', len(images2))


# plt.rcParams.update({'font.size': 16})
# fig, ax = plt.subplots()
# ax.bar('Positive', len(images1), color='black')
# ax.bar('Negative', len(images2), color='black')
# plt.tight_layout()
# plt.show()
# fig.savefig('img.png')

# exibir apenas 2x5 imagens e salvar com positive e negative alternados 
# fig, ax = plt.subplots(2, 5, figsize=(15, 10))
# for i in range(1):
#     for j in range(5):
#         negative in next row
#         img = cv2.imread(os.path.join(path1, images1[i*5 + j]))
#         ax[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         ax[i, j].axis('off')
#         ax[i, j].set_title('Positive')

#         img = cv2.imread(os.path.join(path2, images2[(i+1)*5 + j]))
#         ax[i+1, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         ax[i+1, j].axis('off')
#         ax[i+1, j].set_title('Negative')
      
# plt.tight_layout()
# plt.show()
# fig.savefig('dataset.png')



