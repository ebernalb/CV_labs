#####################################################################################
#
# Computer vision - Laboratory 2
#
# Title: Lab2_740233_736052
#
# Date: 27 October 2020
#
#####################################################################################
#
# Authors: Jorge Condor Lacambra, Edurne Bernal Berdún
#
# Version: 1924629273.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

def indexMatrixToMatchesList(matchesList):

    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):

    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):

    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)

        if dist[indexSort[0]] < minDist:
            if dist[indexSort[0]] < dist[indexSort[1]] * distRatio:
                matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])


    return matches
def match(desc1, desc2, minDist):

    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)

        if dist[indexSort[0]] < minDist:
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])


    return matches

def matchWithEpipolar(desc1, desc2,kp_1,kp_2, distRatio,minDist, distPixel):

    matches = []
    nDesc1 = desc1.shape[0]
    F = matrix_F()

    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)

        if (dist[indexSort[0]] < minDist):
            match = [kDesc1, indexSort[0], dist[indexSort[0]]]
            src = np.float32([kp_1[kDesc1].pt])
            dst = np.float32([kp_2[indexSort[0]].pt])

            x1_pr = np.vstack((src.T, 1))
            x2_pr = np.vstack((dst.T, 1))
            x1_pr = x1_pr[:, 0]
            x2_pr = x2_pr[:, 0]


            l_2 = np.dot(F, x1_pr)

            dist_x2_l2 = np.abs(np.dot(x2_pr, np.dot(F, x1_pr)) / np.sqrt((l_2[0] ** 2 + l_2[1] ** 2)))

            if dist[indexSort[0]] < dist[indexSort[1]] * distRatio:
                matches.append(match)
            elif dist_x2_l2 <= distPixel:
                matches.append(match)


    return matches


def matrix_F():
    wT1 = np.array([[-0.6118, - 0.275, 0.7417, 1.4202],
                    [-0.7909, 0.2285, - 0.5677, 2.4423],
                    [-0.0134, - 0.9339, - 0.3573, 1.2341],
                    [0., 0. ,0. ,1.]])
    wT2 = np.array([[-0.512, - 0.2779, 0.8128, 0.5236],
                    [-0.8586, 0.194, - 0.4745, 1.9537],
                    [-0.0258, - 0.9408, - 0.338, 1.2876],
                    [0. ,0., 0. ,1.]])

    K = np.array([
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0]
    ])

    T_c2_c1 = np.linalg.inv(wT2) @ wT1

    t = np.array([
        [0.0, -T_c2_c1[2, 3], T_c2_c1[1, 3]],
        [T_c2_c1[2, 3], 0.0, -T_c2_c1[0, 3]],
        [-T_c2_c1[1, 3], T_c2_c1[0, 3], 0.0]
    ])
    E = t @ T_c2_c1[0:3, 0:3]
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return F

def find_Homography(ptsource,ptdst):

    A = []
    for i in range(ptsource.shape[0]):
        A.append(np.array([ptsource[0, i], ptsource[1, i], 1.0, 0.0, 0.0, 0.0, -ptdst[0, i] * ptsource[0, i], -ptdst[0, i] * ptsource[1, i], -ptdst[0, i]]))
        A.append(np.array([0.0, 0.0, 0.0, ptsource[0, i], ptsource[1, i], 1.0, -ptdst[1, i] * ptsource[1, i], -ptdst[1, i] * ptsource[0, i], -ptdst[1, i]]))

    A = np.array(A)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    H = V.T[:, -1].reshape((3, 3))

    return H

def find_Homography_RANSAC(source,dst,threshold):
    spFrac = 0.5  # spurious fraction
    P = 0.999 # probability of selecting at least one sample without spurious
    pMinSet = 4  # number of points needed to compute the fundamental matrix

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)
    nAttempts=1000

    matches = np.vstack((source,dst))
    nVotesMax = -1

    for kAttempt in range(nAttempts):
        nVotes = 0

        rng = np.random.default_rng()
        indx_subset = rng.choice(matches.shape[1] - 1, size=pMinSet, replace=False)
        xSubSel = []
        Rmatches = []

        for i in range(matches.shape[1]):
            if i in indx_subset:
                xSubSel.append(matches[:, i])
            else:
                Rmatches.append(matches[:, i])

        xSubSel = np.array(xSubSel).T
        Rmatches = np.array(Rmatches).T

        H_matrix = find_Homography(xSubSel[0:3,:],xSubSel[3:6,:])

        for i in range(Rmatches.shape[1]):

            x1 = Rmatches[0:3, i]
            x2 = Rmatches[3:6, i]

            p_1_h_2 = np.dot(H_matrix, x1)
            p_1_h_2[0] = p_1_h_2[0]/p_1_h_2[2]
            p_1_h_2[1] = p_1_h_2[1]/p_1_h_2[2]

            dist = np.abs( np.linalg.norm(x2 - p_1_h_2) )

            if dist < threshold:
                nVotes = nVotes + 1

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            print(nVotes)
            H_most_voted = H_matrix
            promPoints = xSubSel

    return H_most_voted,promPoints

def find_fundamental_matrix(x0,x1,nx1,ny1,nx2,ny2):
    N0=[
        [1.0 / nx1, 0.0, - 0.5],
        [0.0 , 1.0 / ny1, - 0.5],
        [0.0, 0.0, 1.0]
    ]
    N1 =[
        [1.0 / nx2, 0.0, - 0.5],
        [0.0, 1.0 / ny2, - 0.5],
        [0.0, 0.0, 1.0]
    ]
    T0 = np.linalg.inv(N0) * N0
    T1 = np.linalg.inv(N1) * N1

    A = np.zeros((x0.shape[1], 9))

    for i in range(x0.shape[1]):
        x0[: , i] = np.dot(T0 ,x0[: , i])
        x1[: , i] = np.dot(T1 , x1[: , i])

        A[i, :] = [x0[0, i] * x1[0, i], x0[1, i] * x1[0, i], x1[0, i],
                   x0[0, i] * x1[1, i], x0[1, i] * x1[1, i], x1[1, i],
                   x0[0, i], x0[1, i], 1.0]

    _, _, V = np.linalg.svd(A, full_matrices=True)

    F_norm = V.T[:,-1].reshape((3,3))
    Uf, Sf, Vf = np.linalg.svd(F_norm, full_matrices=True)

    Sf[2] = 0
    F_norm_norm = Uf @ np.diag(Sf) @ Vf
    F  = T1.T @ F_norm_norm @ T0
    return F


def find_fundamental_matrix_RANSAC(source,dst,threshold, nx1, ny1, nx2, ny2):
    spFrac = 0.5  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious
    pMinSet = 8  # number of points needed to compute the fundamental matrix

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac), pMinSet)))
    nAttempts = nAttempts.astype(int)

    matches = np.vstack((source,dst))

    nVotesMax = -1

    for kAttempt in range(nAttempts):
        nVotes = 0
        rng = np.random.default_rng()
        indx_subset = rng.choice(matches.shape[1]-1, size=pMinSet, replace=False)
        xSubSel=[]
        Rmatches=[]

        for i in range(matches.shape[1]):
            if i in indx_subset:
                xSubSel.append(matches[:,i])
            else:
                Rmatches.append(matches[:,i])

        xSubSel = np.array(xSubSel).T
        Rmatches = np.array(Rmatches).T

        F_matrix = find_fundamental_matrix(xSubSel[0:3,:],xSubSel[3:6,:],nx1,ny1,nx2,ny2)

        for i in range(Rmatches.shape[1]):

            x1 = Rmatches[0:3,i]
            x2 = Rmatches[3:6,i]

            l_2 = np.dot(F_matrix,x1)

            dist_x2_l2 = np.abs(np.dot(x2.T,np.dot( F_matrix , x1))/ np.sqrt((l_2[0]**2 + l_2[1]**2)))


            if dist_x2_l2 < threshold:
                nVotes = nVotes + 1

        if nVotes > nVotesMax:
            nVotesMax = nVotes
            print(nVotes)
            pSelected = xSubSel
            F_most_voted = F_matrix

    return F_most_voted, pSelected
if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = timestamp1 + '_undistort.png'
    path_image_2 = timestamp2 + '_undistort.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    ######### EXERCISE 1 ######################################################################

    # Feature extraction
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # Simple matching
    minDist = 90

    matchesList = match(descriptors_1, descriptors_2,minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Plot the first 100 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    ######### EXERCISE 2 ######################################################################

    # Feature extraction
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # NNDR matching
    minDist = 500
    distRatio = 0.8

    matchesList = matchWith2NDRR(descriptors_1, descriptors_2,distRatio, minDist)

    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Plot the first 100 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    ######### EXERCISE 3 ######################################################################

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    color = ['+b', '+g', '+r', '+c', '+m', '+y', '+b', '+g', '+r', '+c','+m', '+y', '+b', '+g', '+r', '+c' ]

    h= find_Homography(x1, x2)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle("Homography: click points in image 1 to translate to image 2")
    ax[0].set_title('Image 1')
    ax[0].imshow(image_pers_1)
    ax[1].set_title('Image 2')
    ax[1].imshow(image_pers_2)

    for i in range(4):
        plt.subplot(ax[0])
        coord_clicked_point = plt.ginput(1, show_clicks=False)
        p_img_1 = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1], 1.0])
        ax[0].plot(p_img_1[0], p_img_1[1], color[i], markersize=15)
        plt.draw()
        p_img_1_homo = h.dot(p_img_1.T)
        ax[1].plot(p_img_1_homo[0] / p_img_1_homo[2], p_img_1_homo[1] / p_img_1_homo[2], color[i], markersize=15)
        plt.draw()
    plt.waitforbuttonpress()

    ######### EXERCISE 4 ######################################################################

    h,x = find_Homography_RANSAC(x1, x2, 1.5)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle("Points to calculate H with RANSAC")
    ax[0].set_title('Image 1')
    ax[0].imshow(image_pers_1)
    ax[1].set_title('Image 2')
    ax[1].imshow(image_pers_2)

    
    for i in range(4):

        plt.subplot(ax[0])
        ax[0].plot(x[0, i] / x[2, i], x[1, i] / x[2, i], '+y', markersize=10)
        ax[1].plot(x[3, i] / x[5, i], x[4, i] / x[5, i], '+y', markersize=10)
        plt.draw()
    plt.waitforbuttonpress()

    plt.suptitle("Click points in image 1 to translate to image 2")
    for i in range(4):
        plt.subplot(ax[0])
        coord_clicked_point = plt.ginput(1, show_clicks=False)
        p_img_1 = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1], 1.0])
        ax[0].plot(p_img_1[0], p_img_1[1], color[i], markersize=15)
        plt.draw()
        p_img_1_homo = h.dot(p_img_1.T)
        ax[1].plot(p_img_1_homo[0] / p_img_1_homo[2], p_img_1_homo[1] / p_img_1_homo[2], color[i], markersize=15)
        plt.draw()
    plt.waitforbuttonpress()

    ######### EXERCISE 5 ######################################################################

    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
    plt.suptitle("Points to calculate F")
    ax2[0].set_title('Image 1')
    ax2[0].imshow(image_pers_1)
    ax2[1].set_title('Image 2')
    ax2[1].imshow(image_pers_2)
    color_lines = ['-b', '-g', '-r', '-c', '-m', '-y', '-b', '-g', '-r', '-c', ]

    f, x = find_fundamental_matrix_RANSAC(x1, x2, 1.5, image_pers_1.shape[1], image_pers_1.shape[0], image_pers_2.shape[1],image_pers_2.shape[0])

    # Calculate and plot epipole from f_RANSAC
    _,_,v= np.linalg.svd(f)
    e0 = v.T[:, -1]
    e0 = e0 / e0[2]
    plt.subplot(ax2[0])
    ax2[1].plot(e0[0], e0[1], 'bo', markersize=3)
    ax2[1].text(e0[0], e0[1], "e0_RANSAC", fontsize=7, color='b')

    # Calculate and plot epipole from f_RANSAC
    _,_,v= np.linalg.svd(matrix_F())
    e0 = v.T[:, -1]
    e0 = e0 / e0[2]
    plt.subplot(ax2[0])
    ax2[1].plot(e0[0], e0[1], 'ro', markersize= 3)
    ax2[1].text(e0[0], e0[1], "e0_GTD", fontsize= 7, color='r')

    for i in range(6):
        plt.subplot(ax2[0])
        coord_clicked_point = plt.ginput(1, show_clicks=False)
        p_img_1 = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1], 1.0])
        ax2[0].plot(p_img_1[0], p_img_1[1], color[i], markersize=15)
        plt.draw()
        l_ep_2 = np.dot(f, p_img_1)

        coor1 = np.array([0, -l_ep_2[2] / l_ep_2[1]])
        coor2 = np.array([image_pers_2.shape[1], (-l_ep_2[2] - image_pers_2.shape[1] * l_ep_2[0]) / l_ep_2[1]])

        plt.subplot(ax2[1])
        ax2[1].plot([coor2[0], coor1[0]], [coor2[1], coor1[1]], color_lines[i], linewidth=2)
        plt.draw()

    plt.waitforbuttonpress()

    ######### EXERCISE 6 ######################################################################

    # Feature extraction
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # NNDR matching
    minDist = 500
    distRatio = 0.6
    distEpipolar = 2.5

    matchesList = matchWithEpipolar(descriptors_1, descriptors_2, keypoints_sift_1, keypoints_sift_2, distRatio, minDist,distEpipolar)

    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    plt.figure(9)
    # Plot the first 100 matches
    imgMatched2 = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched2, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()