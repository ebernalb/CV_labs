#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Laboratory 4
#
# Date: 13 December 2020
#
#####################################################################################
#
# Authors: Jorge Condor Lacambra, Edurne Bernal Berdún
#
# Version: 154197864.0
#
#####################################################################################
#
# Funcions labelled as Legacy are implemented with for loops instead of matrix operations
# and we don't use them in the final version of this code.
#
#####################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.optimize as scOptim
from tqdm import tqdm

def generate_wheel(size):
    """
     Generate wheel optical flow for visualizing colors
     :param size: size of the image
     :return: flow: optical flow for visualizing colors
     """
    rMax = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    u = x - size / 2
    v = y - size / 2
    r = np.sqrt(u ** 2 + v ** 2)
    u[r > rMax] = 0
    v[r > rMax] = 0
    flow = np.dstack((u, v))

    return flow
def draw_hsv(flow, scale):
    """
    Draw optical flow data (Middlebury format)
    :param flow: optical flow data in matrix
    :return: scale: scale for representing the optical flow
    adapted from https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * scale, 255)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return rgb

def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def interpolation(x, y, u1, u2, image, kernel):

    x1 = int(np.floor(x + u1))
    y1 = int(np.floor(y + u2))
    Dx1 = (x + u1) - x1
    Dy1 = (y + u2) - y1
    if (Dx1 == 0 and Dy1 ==0): return get_patch(image, x1, y1, kernel)
    else:
        N = kernel[0]*kernel[1]
        Dx1exp = np.reshape(np.repeat([Dx1],N),(N,1,1))
        Dy1exp = np.reshape(np.repeat([Dy1],N),(N,1,1))
        Dx = np.dstack((1 - Dx1exp, Dx1exp))
        Dy = np.hstack((1 - Dy1exp, Dy1exp))
        I_00 = np.reshape(get_patch(image, x1,  y1,kernel).flatten(),(N,1,1))
        I_01 = np.reshape(get_patch(image, x1, y1+1,kernel).flatten(),(N,1,1))
        I_10 = np.reshape(get_patch(image, x1+1, y1,kernel).flatten(),(N,1,1))
        I_11 = np.reshape(get_patch(image, x1+1, y1+1,kernel).flatten(),(N,1,1))
        
        I_t = np.dstack((np.hstack((I_00,I_10)),np.hstack((I_01,I_11)))) 

        b_t = Dx @ I_t @ Dy

        return np.resize(b_t,(kernel[0],kernel[1]))

def interpolation_legacy(x, y, u1, u2, image):
    x1 = int(np.floor(x + u1))
    y1 = int(np.floor(y + u2))
    Dx1 = (x + u1) - x1
    Dy1 = (y + u2) - y1 
    if (Dx1 == 0 and Dy1 ==0): return image[x1, y1]
    
    else:
        Dx = np.array([1 - Dx1, Dx1])
        Dy = np.array([[1 - Dy1], [Dy1]])
        
        I = np.array([
            [image[x1, y1], image[x1, y1 + 1]],
            [image[x1 + 1, y1], image[x1 + 1, y1 + 1]]
        ])

    return (Dx @ I @ Dy)[0]
   
def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix

    adapted from https://github.com/liruoteng/OpticalFlowToolkit/

    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d

def get_patch(imgChannel, x, y, kernel):
    """
    Apply a convolution kernel to a pixel (no padding considerations)

    :param imgChannel: image input channel
    :param x: x coordinate of point to apply the convolution
    :param y: y coordinate of point to apply the convolution
    :param kernel: convolution kernel
    :return: convolution value at the pixel x,y
    """
    iImgIni = np.int(x - np.floor(kernel[0] / 2))  # the size of the kernel must be odd
    iImgEnd = np.int(x + np.floor(kernel[0] / 2))  # the size of the kernel must be odd
    jImgIni = np.int(y - np.floor(kernel[1] / 2))  # the size of the kernel must be odd
    jImgEnd = np.int(y + np.floor(kernel[1] / 2))  # the size of the kernel must be odd
    imgPatch = imgChannel[iImgIni:iImgEnd + 1, jImgIni:jImgEnd + 1]
    return imgPatch

def crossCorrelation(u_list, image1, image2, kernel, point):    
    u_op = u_list[0]
    Encc=-1
    MaxEncc = -9999
    I0 = get_patch(image1,point[1],point[0],kernel)
    for u in u_list:
        I1 = get_patch(image2, point[1]+u[0],point[0]+u[1],kernel)

        I0_mean = np.mean(I0)
        I1_mean = np.mean(I1)
        er_I0 = I0 - I0_mean
        er_I1 = I1 - I1_mean
        numer = np.sum(np.multiply(er_I0,er_I1))
        denom = np.sqrt(np.sum(np.multiply(er_I0,er_I0)) * np.sum(np.multiply(er_I1,er_I1)))

        if (denom>0): 
            Encc = numer/denom
            if (Encc > MaxEncc):
               u_op = u
               MaxEncc = Encc
                   
      
    return u_op

def central_diff(image,x,y,kernel):

    Ix_patch = (np.float32(get_patch(image, x+1, y, kernel)) - np.float32(get_patch(image, x-1, y, kernel))) / 2.0
    Iy_patch = (np.float32(get_patch(image, x, y+1, kernel)) - np.float32(get_patch(image, x, y-1, kernel))) / 2.0
    
    return Ix_patch, Iy_patch

def central_diff_single_pixel(image,x,y):
 
    Ix = (float(image[x+1,y]) - float(image[x-1,y]))/2.0
    Iy = (float(image[x,y+1]) - float(image[x, y-1]))/2.0
    
    return np.array([Ix,Iy])
def compute_A_legacy(window,img1):
    vector = []
    EIx_Ix,EIx_Iy,EIy_Iy = 0,0,0
    for i in window[0]:
        for j in window[1]:
            vector = central_diff_single_pixel(img1,i,j)
            EIx_Ix = EIx_Ix + vector[0]*vector[0]
            EIx_Iy = EIx_Iy + vector[0]*vector[1]
            EIy_Iy = EIy_Iy + vector[1]*vector[1]
    
    A = np.array([
        [EIx_Ix, EIx_Iy],
        [EIx_Iy, EIy_Iy]
        ])
    return A
def compute_A(Ix_patch,Iy_patch):

    Ix2 = np.sum(np.multiply(Ix_patch,Ix_patch))
    Ixy = np.sum(np.multiply(Ix_patch,Iy_patch))
    Iy2 = np.sum(np.multiply(Iy_patch,Iy_patch))
    A = np.array([
        [Ix2,Ixy],
        [Ixy,Iy2]])
    return A

def compute_b(point, img1, img2, u, kernel, window, I0,Ix,Iy):
    I1 = interpolation(point[1], point[0],u[0],u[1], img2, kernel)
    It = np.float32(I1) - np.float32(I0)

    b = np.array([
        [- np.sum(np.multiply(Ix,It))],
        [- np.sum(np.multiply(Iy,It))]
         ])

    return b

def compute_b_legacy(img1, img2, u, window):
    vector = []
    EIx_It,EIy_It =0,0

    for i in window[0]:
        for j in window[1]:
            vector = central_diff_single_pixel(img1, i, j)
            I0 = img1[i, j]
            #I1 = bilin_inter(np.array([i+u[0], j+ u[1]]), img2)
            I1 = interpolation_legacy(i, j, u[0], u[1], img2)

            It = float(I1) - float(I0)
            EIx_It = EIx_It + vector[0] * It
            EIy_It = EIy_It + vector[1] * It
    return  np.array([[-EIx_It],[-EIy_It]])

def lucas_kanade(img1, img2, u, point, window, kernel):
    
    epsilon = 0.005
    Ix,Iy = central_diff(img1,point[1],point[0],kernel)
    #A = compute_A_legacy(window, img1)
    A = compute_A(Ix,Iy)

    if (np.linalg.det(A)<=0.1):
        print("Impossible to Invert A. Returning Original U aproximation")
        return u

    invA = np.linalg.inv(A)

    var_u = np.array([10,10])
    original_u = u
    I0 = get_patch(img1, point[1],point[0],kernel)
    iter=0
    while (np.linalg.norm(var_u) >= epsilon):
        b = compute_b(point, img1, img2, u, kernel, window, I0,Ix,Iy)
        #b = compute_b_legacy(img1, img2, u, window)
        var_u = invA @ b
        u = u + np.array([var_u[0,0],var_u[1,0]])
        if (u[0]<-99 or u[0]>99 or u[1]<-99 or u[1]> 99): 
            return np.array([original_u[1],original_u[0]])
        iter+=1
        if (iter>=100): 
           return np.array([original_u[1],original_u[0]]) #np.array([u[1],u[0]])
    return np.array([u[1],u[0]])

if __name__ == '__main__':

    image_1 = read_image('frame10.png')
    image_2 = read_image('frame11.png')
    printable_image_1 = cv.imread('frame10.png')
    printable_image_1 = cv.cvtColor(printable_image_1, cv.COLOR_BGR2RGB)
    printable_image_2 = cv.imread('frame11.png')
    printable_image_2 = cv.cvtColor(printable_image_2, cv.COLOR_BGR2RGB)


    # List of sparse points
    points_selected = np.uint16(np.loadtxt('points_selected.txt'))

    flow_12 = read_flo_file("flow10.flo", verbose=True)
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(np.float)
    
    unknownFlowThresh = 1e9
    binUnknownFlow = flow_12 > unknownFlowThresh

    #########################################################################################
    ##                          Sparse-Optical Flow                                        ##
    #########################################################################################
    
    
    Possible_u0 = np.arange(-5.0, 5.0, 1)
    Possible_u1 = np.arange(-5.0, 5.0, 1)

    Possible_u = []

    for i in Possible_u0:
        for j in Possible_u1:
            Possible_u.append(np.array([i,j]))
    kernel = np.array([11,11])
    flow_est_sparse = np.zeros((6,2))

    for i,point in enumerate(points_selected):
        window_x = range(point[1]-5,point[1]+6,1)
        window_y = range(point[0]-5,point[0]+6,1)
        window = [window_x, window_y]

        u = crossCorrelation(Possible_u, image_1, image_2, kernel, point )
        print('Point:',point,'    ', 'U:', np.array([u[1],u[0]]))
        u_LK = lucas_kanade(image_1, image_2, u, point, window, kernel)

        print('Point:',point,'    ', 'U_LK:', u_LK)
        flow_est_sparse[i,:] = u_LK
      
    unknownFlowThresh = 1e9
    binUnknownFlow = flow_gt > unknownFlowThresh

    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    
    error_sparse[binUnknownFlow] = 0
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))
        
    # Plot results for sparse optical flow
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(printable_image_1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow')
    axs[1].imshow(printable_image_1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error with respect to GT')
    plt.show()

    #########################################################################################
    ##                          Dense - OpticalFlow                                        ##
    #########################################################################################
    unknownFlowThresh = 1e9
    binUnknownFlow = flow_12 > unknownFlowThresh

    padding_size = 100
    image_1_pad = np.pad(image_1,padding_size,mode='edge')
    image_2_pad = np.pad(image_2,padding_size,mode='edge')
    padding_multiple_dim = ((padding_size,padding_size),(padding_size,padding_size),(0,0))

    printable_image_1_pad = np.pad(printable_image_1, pad_width=padding_multiple_dim, mode='edge')
    printable_image_2_pad = np.pad(printable_image_2, pad_width=padding_multiple_dim, mode='edge')
    
    flow_12_padded = np.pad(flow_12, pad_width=padding_multiple_dim)
    #flow_12_padded = np.zeros((image_1_pad.shape[0],image_1_pad.shape[1],2))
    #flow_12_padded[:,:,0] = np.pad(flow_12[:,:,0],(padding_size,padding_size))
    #flow_12_padded[:,:,1] = np.pad(flow_12[:,:,1],(padding_size,padding_size))
    
    size_patch_x = 562
    size_patch_y = 366
    px = np.uint16(np.floor(image_1.shape[1]/2)+padding_size) #100
    py = np.uint16(np.floor(image_1.shape[0]/2)+padding_size) #150
    """
    
    size_patch_x = 50
    size_patch_y = 50
    px = 100+100 #100
    py = 150+100 #150
    """
    patch_x_start = px - int(size_patch_x/2)
    patch_x_finish = px + int(size_patch_x/2)
    patch_y_start = py - int(size_patch_y/2)
    patch_y_finish = py + int(size_patch_y/2)

    patch_x = range(patch_y_start,patch_y_finish,1)
    patch_y = range(patch_x_start,patch_x_finish,1)
    patch = [patch_x, patch_y]

    flow_est_dense = np.ones((len(patch_x),len(patch_y),2))
    kernel = np.array([11,11])
    i=0
    '''
    with tqdm(range(len(patch_x) * len(patch_y)), ascii=True) as pbar:
        for i,x in enumerate(patch[0]):
            for j,y in enumerate(patch[1]):
               window_x = range(x-5,x+6,1)
               window_y = range(y-5,y+6,1)
               window = [window_x, window_y]

               u = crossCorrelation(Possible_u, image_1_pad, image_2_pad, kernel, np.array([y,x]) )
               u_LK = lucas_kanade(image_1_pad, image_2_pad, u,  np.array([y,x]), window, kernel)

               flow_est_dense[i,j,:] = u_LK
               pbar.update(1)
          
           #print('i',i,'j',j,' Uk: ',u_LK,' TRUE: ',flow_12[i,j,:])
    np.save('flow_minipatch.npy',flow_est_dense)   
    '''
    ## Dense optical flow
    flow_est_dense=np.load('flow_all_image_LucasKanade.npy')
    flow_error = flow_est_dense - flow_12_padded[patch_y_start:patch_y_finish,patch_x_start:patch_x_finish,:]
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))

    # Plot results for dense optical flow
    scale = 40
    wheelFlow = generate_wheel(256)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(printable_image_1_pad[patch_y_start:patch_y_finish,patch_x_start:patch_x_finish], cmap='gray')
    axs[0, 0].title.set_text('image 1')
    axs[1, 0].imshow(printable_image_2_pad[patch_y_start:patch_y_finish,patch_x_start:patch_x_finish], cmap='gray')
    axs[1, 0].title.set_text('image 2')
    axs[0, 1].imshow(draw_hsv(flow_12_padded[patch_y_start:patch_y_finish,patch_x_start:patch_x_finish,:], scale))
    axs[0, 1].title.set_text('Optical flow ground truth')
    axs[1, 1].imshow(draw_hsv(flow_est_dense, scale))
    axs[1, 1].title.set_text('LK estimated optical flow ')
    axs[0, 2].imshow(error_norm, cmap='jet')
    axs[0, 2].title.set_text('Optical flow error norm')
    axs[1, 2].imshow(draw_hsv(wheelFlow, 3))
    axs[1, 2].title.set_text('Color legend')
    axs[1, 2].set_axis_off()
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    
    #########################################################################################
    ##                          Variational Dense -OpticalFlow                              ##
    #########################################################################################
    
    Op_variational = cv.createOptFlow_DualTVL1()
    flow_variational = Op_variational.calc(image_1,image_2,None)

     ## Dense optical flow
    flow_error = flow_variational - flow_12
    flow_error[binUnknownFlow] = 0
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))

    # Plot results for dense optical flow
    scale = 40
    wheelFlow = generate_wheel(256)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(printable_image_1)
    axs[0, 0].title.set_text('image 1')
    axs[1, 0].imshow(printable_image_2)
    axs[1, 0].title.set_text('image 2')
    axs[0, 1].imshow(draw_hsv(flow_12 * np.bitwise_not(binUnknownFlow), scale))
    axs[0, 1].title.set_text('Optical flow ground truth')
    axs[1, 1].imshow(draw_hsv(flow_variational, scale))
    axs[1, 1].title.set_text('Variational estimated optical flow ')
    axs[0, 2].imshow(error_norm, cmap='jet')
    axs[0, 2].title.set_text('Optical flow error norm')
    axs[1, 2].imshow(draw_hsv(wheelFlow, 3))
    axs[1, 2].title.set_text('Color legend')
    axs[1, 2].set_axis_off()
    fig.subplots_adjust(hspace=0.5)
    plt.show()

    #########################################################################################
    ##                          Farneback -OpticalFlow                                     ##
    #########################################################################################
    
    flow_Farneback = cv.calcOpticalFlowFarneback(image_1,image_2, flow=None,
                                      pyr_scale=0.5, levels=5, winsize=13,
                                      iterations=10, poly_n=5, poly_sigma=1.1,
                                      flags=0)

    ## Dense optical flow
    flow_error = flow_Farneback - flow_12
    flow_error[binUnknownFlow] = 0
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))


    # Plot results for dense optical flow
    scale = 40
    wheelFlow = generate_wheel(256)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(printable_image_1)
    axs[0, 0].title.set_text('image 1')
    axs[1, 0].imshow(printable_image_2)
    axs[1, 0].title.set_text('image 2')
    axs[0, 1].imshow(draw_hsv(flow_12 * np.bitwise_not(binUnknownFlow), scale))
    axs[0, 1].title.set_text('Optical flow ground truth')
    axs[1, 1].imshow(draw_hsv(flow_Farneback, scale))
    axs[1, 1].title.set_text('Farneback estimated optical flow ')
    axs[0, 2].imshow(error_norm, cmap='jet')
    axs[0, 2].title.set_text('Optical flow error norm')
    axs[1, 2].imshow(draw_hsv(wheelFlow, 3))
    axs[1, 2].title.set_text('Color legend')
    axs[1, 2].set_axis_off()
    fig.subplots_adjust(hspace=0.5)
    plt.show()