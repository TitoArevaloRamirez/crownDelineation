import numpy as np
from PIL import Image
import cv2
import tensorly as tl

from tensorly.decomposition import tucker

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

from skimage.segmentation import slic
from skimage.color import label2rgb

from skimage.measure import label, regionprops


from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


from skimage.metrics import structural_similarity as ssim


from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte


from skimage import io, color


def normalizeTucker(data_tucker):
    T, H, W, B = data_tucker.shape
    Xn = data_tucker.copy().astype(np.float64)
    
    for b in range(B):
        for t in range(T):
            channel = Xn[t,:, :, b]
            Xn[t, :, :, b] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    return Xn


def main():
    root = "/Users/titoarevalo-ramirez/Data/Talca2025/Registered/Darwin/"
    
    b1 = np.array(Image.open(root + "b_2020_11_21_1.tif"))
    b2 = np.array(Image.open(root + "b_2020_11_21_2.tif"))
    b3 = np.array(Image.open(root + "b_2020_11_22_1.tif"))
                                                          
    g1 = np.array(Image.open(root + "g_2020_11_21_1.tif"))
    g2 = np.array(Image.open(root + "g_2020_11_21_2.tif"))
    g3 = np.array(Image.open(root + "g_2020_11_22_1.tif"))
                                                          
    r1 = np.array(Image.open(root + "r_2020_11_21_1.tif"))
    r2 = np.array(Image.open(root + "r_2020_11_21_2.tif"))
    r3 = np.array(Image.open(root + "r_2020_11_22_1.tif"))

    rEd1 = np.array(Image.open(root + "rEd_2020_11_21_1.tif"))
    rEd2 = np.array(Image.open(root + "rEd_2020_11_21_2.tif"))
    rEd3 = np.array(Image.open(root + "rEd_2020_11_22_1.tif"))
                                                              
    nir1 = np.array(Image.open(root + "nir_2020_11_21_1.tif"))
    nir2 = np.array(Image.open(root + "nir_2020_11_21_2.tif"))
    nir3 = np.array(Image.open(root + "nir_2020_11_22_1.tif"))  

    [n, m] = np.shape(b1)
    b1_r = b1.reshape((1,n, m))
    b2_r = b2.reshape((1,n, m))
    b3_r = b3.reshape((1,n, m))

    g1_r = g1.reshape((1,n, m))
    g2_r = g2.reshape((1,n, m))
    g3_r = g3.reshape((1,n, m))

    r1_r = r1.reshape((1,n, m))
    r2_r = r2.reshape((1,n, m))
    r3_r = r3.reshape((1,n, m))

    rEd1_r = rEd1.reshape((1,n, m))
    rEd2_r = rEd2.reshape((1,n, m))
    rEd3_r = rEd3.reshape((1,n, m))

    nir1_r = nir1.reshape((1,n, m))
    nir2_r = nir2.reshape((1,n, m))
    nir3_r = nir3.reshape((1,n, m))

    b_t = np.concatenate((b1_r, b2_r, b3_r), axis=0)
    g_t = np.concatenate((g1_r, g2_r, g3_r), axis=0)
    r_t = np.concatenate((r1_r, r2_r, r3_r), axis=0)
    rEd_t = np.concatenate((rEd1_r, rEd2_r, rEd3_r), axis=0)
    nir_t = np.concatenate((nir1_r, nir2_r, nir3_r), axis=0)

    data_np = np.stack((b_t, g_t, r_t, rEd_t, nir_t), axis=3)

    data_tucker = data_np[:,1024:1532,1024:1532,:]
    Xn = normalizeTucker(data_tucker)

    print(Xn.shape)
    tensor = tl.tensor(Xn, dtype=tl.float64)
    core, factors = tucker(tensor, rank=[1, 220, 220, 5], verbose=2)

    X_hat = tl.tucker_to_tensor((core, factors))  # reconstructed tensor

    print(np.max(X_hat))

    Xn_hat = normalizeTucker(X_hat)

    print(np.max(Xn_hat))

    b_scaled = Xn_hat[0, :, :, 0]
    g_scaled = Xn_hat[0, :, :, 1]
    r_scaled = Xn_hat[0, :, :, 2]
    rEd_scaled = Xn_hat[0, :, :, 3]
    nir_scaled = Xn_hat[0, :, :, 4]

    ndvi = (nir_scaled - r_scaled )/(r_scaled + nir_scaled + 0.0001)
    ndvi_scaled = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))

    ndvi_raw = (Xn[0,:,:,4] - Xn[0,:,:,2] )/(Xn[0,:,:,4] + Xn[0,:,:,2] + 0.0001)
    ndvi_raw = (ndvi_raw - np.min(ndvi_raw)) / (np.max(ndvi_raw) - np.min(ndvi_raw))

    #image = np.stack((b_scaled, g_scaled, r_scaled, rEd_scaled, nir_scaled), 2)
    bgr = np.stack((b_scaled, g_scaled, r_scaled), 2)
    cir = np.stack((nir_scaled, r_scaled, g_scaled), 2)
    ndvi = np.stack((ndvi_scaled, ndvi_scaled, ndvi_scaled), 2)

    cir_raw = np.stack((Xn[1,:,:,4], Xn[1,:,:,1], Xn[1,:,:,2]), 2)


    img = np.asarray(255*cir, dtype=np.uint8)
    img = img_as_ubyte(img)

    print(np.max(img))

    # Convert images to grayscale
    img1_gray = rgb2gray(cir_raw )
    img2_gray = rgb2gray(cir)

    # Ensure images have the same dimensions (if necessary, though they should match)
    # If dimensions differ, you may need to resize.

    # Calculate SSIM and the difference map
    # data_range specifies the range of the image data (e.g., 1.0 for float images)
    s_index, diff = ssim(img1_gray, img2_gray, full=True, data_range=img2_gray.max() - img2_gray.min())
    
    # The diff image is a float array, we convert it to uint8 for display
    # by rescaling the difference map to the 0-255 range and making it an absolute value
    diff = (diff * 255).astype("uint8")

    print(f"SSIM score: {s_index}")
    
    # Visualize the images and the difference
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes.ravel()

    ax[0].imshow(img1_gray)
    ax[0].set_title("Image 1")
    ax[0].axis('off')

    ax[1].imshow(img2_gray)
    ax[1].set_title("Image 2")
    ax[1].axis('off')

    # Darker regions in the difference image indicate areas of dissimilarity
    ax[2].imshow(diff, cmap='gray')
    ax[2].set_title("Difference Map")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


    #labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


    vectorized_img = img.reshape((-1,3))
    vectorized_img= np.float32(vectorized_img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 3
    attempts=10
    ret,labels,center=cv2.kmeans(vectorized_img,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)


    print(labels.shape)
    print(np.unique(labels))
    print(center)

    center = np.uint8(center)
    center[0,:] = 0
    center[1,:] = 255
    center[2,:] = 0
    res = center[labels.flatten()]
    result_image = res.reshape((img.shape))

    #print(result_image.shape)
    kernel = np.ones((5,5),np.uint8)
    result_image = cv2.morphologyEx(result_image, cv2.MORPH_CLOSE, kernel)
    #result_image = -(result_image - 255)
    kernel = np.ones((5,5),np.uint8)
    result_image = cv2.erode(result_image[:,:,1],kernel,iterations = 1)

    label_image = label(result_image)

    label_image[label_image!=1] = 0

    distance = ndi.distance_transform_edt(Xn[1,:,:,1]*label_image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=label_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(ndvi[:,:,0], markers, mask=label_image)
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(Xn[1,:,:,1]*label_image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    fig.tight_layout()
    plt.show()



    image_label_overlay = label2rgb(label_image, image=bgr, bg_label=0)

    print(np.asarray(label_image).shape)


    contours,hierarchy = cv2.findContours(img_as_ubyte(label_image), cv2.RETR_EXTERNAL, 2)
     
    cnt = contours[0]
    M = cv2.moments(cnt)
    print(cnt)
    print(contours)


    cv2.drawContours(image=bgr, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.imshow(ndvi, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(image_label_overlay)
    plt.axis('off')
    plt.subplot(3, 1, 3)
    plt.imshow(bgr)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    main()



