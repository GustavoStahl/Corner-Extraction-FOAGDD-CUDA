import numpy as np
import cv2
from scipy import signal

def assert_arrays(array_name, array_new):
    array_prev = np.load(f"gt_arrays/{array_name}.npy", allow_pickle=True)
    array_diff = np.abs(array_prev - array_new)
    assert array_diff.max() <= 1e-5, f"{array_name} don't match. Min {array_diff.min()}, Max {array_diff.max()}"
    print(f"{array_name} match. Min {array_diff.min()}, Max {array_diff.max()}")

def nonma(cim,threshold,radius):
    rows,cols=np.shape(cim)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    mx=cv2.dilate(cim,kernel)
    bordermask=np.zeros_like(cim,dtype=np.int)
    bordermask[radius+1:rows-radius,radius+1:cols-radius]=1
    t = (cim == mx).astype(int)
    t2 = (cim > threshold).astype(int)
    cimmx=t & t2 & bordermask
    return np.array(cimmx.nonzero()).T

def foggdd(im, threshold):
    rho = 1.5
    directions_n = 8
    sigmas = np.linspace(1.5, 4.5, num=3)
    eps = 2.220 * 10 ** (-16) # 2.22e-16 
    nonma_radius = 5

    if im.shape[2]!=1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows, cols = im.shape[:2]

    patch_size = 7
    im_padded = cv2.copyMakeBorder(im, patch_size, patch_size, patch_size, patch_size, cv2.BORDER_REFLECT).astype(np.float32)

    lattice_size = 31 # consider the origin in the lattice
    lattice_axis = np.arange(-(lattice_size//2), lattice_size//2+1, dtype=int)
    lattice_xx, lattice_yy = np.meshgrid(lattice_axis, lattice_axis, indexing="ij")
    lattice = np.concatenate((np.expand_dims(lattice_xx, -1), np.expand_dims(lattice_yy,-1)), axis=-1)

    rho_mat = np.array([rho, 0, 0, 1/rho]).reshape(2,2)

    templates = np.empty((directions_n, len(sigmas), *im_padded.shape[:2]), dtype=float)
    anigs_directions = np.empty((directions_n, len(sigmas), lattice_size, lattice_size), dtype=float)
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
            anigs_directions[direction_idx, sigma_idx] = anigs_direction

            conv_filter = anigs_direction - anigs_direction.sum()/anigs_direction.size
            template = cv2.filter2D(im_padded, -1, cv2.flip(conv_filter,-1), borderType=cv2.BORDER_CONSTANT) 
            # template = signal.convolve2d(im_padded, conv_filter, mode='same')
            templates[direction_idx, sigma_idx] = template

    # assert_arrays("anigs_directions", anigs_directions)
    # assert_arrays("templates", templates) #! test
    # templates = np.load("gt_arrays/templates.npy", allow_pickle=True)

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

    # another approach
    # mask = [20,27,34,12,19,26,33,40,4,11,18,25,32,39,46,3,10,17,24,31,38,45,2,9,16,23,30,37,44,8,15,22,29,36,14,21,28]

    measure = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        top = i + patch_size - 3
        bottom = i + patch_size + 3
        for j in range(cols):
            left = j + patch_size - 3
            right = j + patch_size + 3

            templates_slice = templates[:, 0, top:bottom+1, left:right+1][:, mask] # 8x37
            templates_slice = np.abs(templates_slice)
            mat = templates_slice @ templates_slice.T
            measure[i, j] = np.linalg.det(mat) / (np.trace(mat) + eps)
            
    # assert_arrays("measure", measure) #! test

    measure_nonma = nonma(measure, threshold, nonma_radius)
    # assert_arrays("measure_nonma", measure_nonma)

if __name__ == "__main__":
    im = cv2.imread("17.bmp")
    foggdd(im, 10 ** 8.4)   