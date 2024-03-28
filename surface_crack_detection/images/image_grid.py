import matplotlib.pyplot as plt
import cv2 
import os


def show_images(original_images, generated_masks, annotated_images, unet_images):
    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()

    fig, axs = plt.subplots(4, 5, figsize=(10, 10))

    for i in range(5):
        axs[0, i].imshow(original_images[i])
        axs[0, i].axis('off')

        axs[1, i].imshow(generated_masks[i], cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title(titles_sam_mask[i])


        axs[2, i].imshow(annotated_images[i])
        axs[2, i].axis('off')
        axs[2, i].set_title(titles_sam_mask[i])
        
        axs[3, i].imshow(unet_images[i], cmap='gray')
        axs[3, i].axis('off')
        axs[3, i].set_title(titles_unet[i])

    fig.text(0.1, 0.8, 'Original', va='center', rotation='vertical')
    fig.text(0.1, 0.6, 'SAM', va='center', rotation='vertical')
    fig.text(0.1, 0.4, 'SAM Mask', va='center', rotation='vertical')
    fig.text(0.1, 0.2, 'U-Net', va='center', rotation='vertical')
    
    fig.savefig('best_images/image_grid.eps', format='eps', dpi=300, bbox_inches='tight')
    
    plt.show()
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is not None:
            images.append(img)
    return images

original_images_path = 'best_images/original/'
genereted_masks_path = 'best_images/binary/'
annotated_images_path = 'best_images/segmented/'
unet_masks_path = 'best_images/u_net/'

original_images = load_images_from_folder(original_images_path)
generated_masks = load_images_from_folder(genereted_masks_path)
annotated_images = load_images_from_folder(annotated_images_path)
unet_images = load_images_from_folder(unet_masks_path)

metrics_sam_mask = [0.9635, 0.9541, 0.9423, 0.9401, 0.9415]
titles_sam_mask = ['IoU: {:.4f}'.format(sam) for sam in metrics_sam_mask]

metrics_unet = [0.8400, 0.8544, 0.8307, 0.8072, 0.4781]
titles_unet = ['IoU: {:.4f}'.format(sam) for sam in metrics_unet]



show_images(original_images, annotated_images, generated_masks, unet_images)