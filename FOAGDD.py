import sys
import numpy as np
from time import time
import cv2

def nonma(cim, threshold, radius):
    rows, cols = cim.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    mx = cv2.dilate(cim, kernel)
    bordermask = np.zeros_like(cim, dtype=bool)
    bordermask[radius+1:rows-radius, radius+1:cols-radius] = True
    t = cim == mx
    t2 = cim > threshold
    cimmx = t & t2 & bordermask
    return np.transpose(cimmx.nonzero())

def compute_templates(image, directions_n, sigmas, rho, lattice_size):    
    rho_mat = np.array([rho, 0, 0, 1/rho]).reshape(2,2)

    lattice_axis = np.arange(-(lattice_size//2), lattice_size//2+1, dtype=int)
    lattice_xx, lattice_yy = np.meshgrid(lattice_axis, lattice_axis, indexing="ij")
    lattice = np.concatenate((np.expand_dims(lattice_xx, -1), np.expand_dims(lattice_yy,-1)), axis=-1)

    templates = np.empty((directions_n, len(sigmas), *image.shape[:2]), dtype=float)
    for direction_idx in range(directions_n):
        theta = direction_idx * np.pi / directions_n
        R = np.array([np.cos(theta), np.sin(theta), -np.sin(theta), np.cos(theta)]).reshape(2,2)
        R_T = R.T

        for sigma_idx, sigma in enumerate(sigmas):

            anigs_direction = np.empty((lattice_size, lattice_size), dtype=float) 
            for lattice_index in np.ndindex(lattice.shape[:-1]):
                n = lattice[lattice_index].reshape(-1,1) # [nx, ny].T
                n_T = n.T # [nx, ny]

                agk = 1/(2 * np.pi * sigma * sigma) * np.exp(-1/(2 * sigma) * n_T @ R_T @ rho_mat @ R @ n)
                agdd = -rho * np.array([np.cos(theta), np.sin(theta)]) @ n * agk  
                anigs_direction[lattice_index] = agdd

            conv_filter = anigs_direction - anigs_direction.sum()/anigs_direction.size

            template = cv2.filter2D(image, -1, cv2.flip(conv_filter,-1), borderType=cv2.BORDER_CONSTANT) 
            # template = signal.convolve2d(im_padded, conv_filter, mode='same')
            templates[direction_idx, sigma_idx] = np.abs(template)
    return templates            

def foagdd(_im, threshold):
    im = _im.copy()

    rho = 1.5
    directions_n = 8
    sigmas = np.linspace(1.5, 4.5, num=3)
    eps = 2.22e-16 
    nonma_radius = 5

    if im.ndim >= 3 and im.shape[2] != 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows, cols = im.shape[:2]

    patch_size = 7
    im_padded = cv2.copyMakeBorder(im, patch_size, patch_size, patch_size, patch_size, cv2.BORDER_REFLECT).astype(np.float32)

    lattice_size = 31 # consider the origin in the lattice
    templates = compute_templates(im_padded, directions_n, sigmas, rho, lattice_size)

    # NOTE: The code below is creating the following mask
    # ┌───────┐
    # │0011100│
    # │0111110│
    # │1111111│
    # │1111111│
    # │1111111│
    # │0111110│
    # │0011100│
    # └───────┘

    mask = np.ones((patch_size,patch_size), dtype=bool)
    mask[0,:2] = False
    mask[0,-2:] = False
    mask[1,0] = False
    mask[1,-1] = False
    mask[-2,0] = False
    mask[-2,-1] = False
    mask[-1,:2] = False
    mask[-1,-2:] = False

    corner_measure = np.empty((rows, cols), dtype=float)
    for (i,j) in np.ndindex(rows,cols):
        top = i + patch_size - 3
        bottom = i + patch_size + 3
        left = j + patch_size - 3
        right = j + patch_size + 3

        templates_slice = templates[:, 0, top:bottom+1, left:right+1] # 8x49
        templates_slice = templates_slice[:, mask] # 8x37
        #NOTE this matrix is symmetric, thus it has real eigenvalues and eigenvectors
        mat = templates_slice @ templates_slice.T
        #NOTE approximation of: product of eigenvalues / sum of eigenvalues
        corner_measure[i, j] = np.linalg.det(mat) / (np.trace(mat) + eps) 
            
    points_of_interest = nonma(corner_measure, threshold, nonma_radius)

    for sigma_idx in range(1, len(sigmas)):
        poi_maintained_mask = []
        for (i,j) in points_of_interest:

            top = i + patch_size - 3
            bottom = i + patch_size + 3
            left = j + patch_size - 3
            right = j + patch_size + 3

            templates_slice = templates[:, sigma_idx, top:bottom+1, left:right+1] # 8x49
            templates_slice = templates_slice[:, mask] # 8x37
            mat = templates_slice @ templates_slice.T
            corner_measure = np.linalg.det(mat) / (np.trace(mat) + eps) 

            poi_maintained_mask.append(corner_measure > threshold)

        points_of_interest = points_of_interest[poi_maintained_mask]
    return points_of_interest        

if __name__ == "__main__":
    num_iters = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    image_path = sys.argv[2] if len(sys.argv) >= 3 else "data/17.bmp"

    im = cv2.imread(image_path)

    time_taken = 0
    for i in range(num_iters):
        start = time()
        poi_arr = foagdd(im, 10 ** 8.4)   
        if i > 0:
            time_taken += time() - start

    if num_iters != 1:
        num_iters -= 1.0

    print(f"Average elapsed time in seconds: {time_taken/(num_iters-1)} s")
    print(f"Points of interest found: {poi_arr.shape}")

    # [[y1,x1],...] --> [[x1,y1],...]
    poi_arr = np.flip(poi_arr,axis=-1)

    for poi in poi_arr:
        color = (255,0,0)
        cv2.drawMarker(im, poi, color, cv2.MARKER_SQUARE, markerSize=2, thickness=1, line_type=cv2.LINE_AA)

    cv2.imwrite("data/result.jpg", im)