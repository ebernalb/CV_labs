
#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 5
#
# Title: Laboratory 5
#
# Date: 4 January 2021
#
#####################################################################################
#
# Authors: Jorge Condor Lacambra, Edurne Bernal Berdún
#
# Version: 154.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
import scipy.optimize as scOptim

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)
def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, lineWidth=lWidth)
def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)
def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)
def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)
def KannalaBrandt_projection( X, D, K):
    
    X /= X[3]
    R = np.sqrt(X[0]**2 + X[1]**2)
    theta = np.arctan2(R, X[2]) #angle between R and Z in the vertical plane
    d_theta = theta + D[0] * theta**3 + D[1] * theta**5 + D[2] * theta**7 + D[3] * theta**9 
    phi = np.arctan2(X[1],X[0]) #angle between X and Y in the image plane
    KB = np.array([
        [d_theta * np.cos(phi)],
        [d_theta * np.sin(phi)],
        [1.0]])
    u = K @ KB
    return u[:, 0]/u[2, 0]
def KannalaBrandt_unprojection( u, D, K):
    Xc = np.linalg.inv(K) @ u
    r = np.sqrt((Xc[0]**2 + Xc[1]**2) / Xc[2]**2)
    poly = [D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -r]
    roots = np.roots(poly)
    theta = 0
    for root in roots:
        if np.isreal(root):
            theta = np.real(root)

    phi = np.arctan2(Xc[1], Xc[0])
    v = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)])
    return v.T
def triangulation(v1, v2, c1Tc2):
    pSym_1 = np.array([[-v1[1]], [v1[0]], [0], [0]])
    pSym_1 = pSym_1 / np.linalg.norm(pSym_1)
    pPar_1 = np.array([[-v1[2]*v1[0]], [-v1[2]*v1[1]], [v1[0]**2 +v1[1]**2], [0]])
    pPar_1 = pPar_1 / np.linalg.norm(pPar_1)
    pSym_2 = np.array([[-v2[1]], [v2[0]], [0], [0]])
    pSym_2 = pSym_2 / np.linalg.norm(pSym_2)
    pPar_2 = np.array([[-v2[2]*v2[0]], [-v2[2]*v2[1]], [v2[0]**2 +v2[1]**2], [0]])
    pPar_2 = pPar_2 / np.linalg.norm(pPar_2)
    pSym_2_proj = c1Tc2.T @ pSym_1
    pPar_2_proj = c1Tc2.T @ pPar_1
    A = np.vstack((pSym_2_proj.T, pPar_2_proj.T, pSym_2.T, pPar_2.T))

    _, _, V = np.linalg.svd(A, full_matrices=True)
    Point_3D = V.T[:, -1]
  
    return Point_3D / Point_3D[3]
def crossMatrixInv(M):
     x = [M[2, 1], M[0, 2], M[1, 0]]
     return x
def crossMatrix(x):
     M = np.array([[0, -x[2], x[1]],
     [x[2], 0, -x[0]],
     [-x[1], x[0], 0]], dtype="object")
     return M
def resBundleProjection_n_View(Op, xData, K1, k2, D1, D2, wTc1, wTc2, nCameras, nPoints):
    
    #Op[0:3] -> watwb: tx,ty,tz
    #Op[3:6] -> waRwb: Rx,Ry,Rz
    
    #Op[6:nPoints+6] -> 3DXx,3DXy,3DXz (respect to C1_B)

    waRwb = linalg.expm(crossMatrix(Op[3:6]))
    watwb = Op[0:3]
    waTwb = np.vstack((np.hstack((waRwb, np.expand_dims(watwb, axis=1))),np.array([0,0,0,1])))

    res = []
    c1aTc1b = np.linalg.inv(wTc1) @ waTwb @ wTc1
    c1Tc2 = np.linalg.inv(wTc1) @ wTc2
    for i in range(nPoints):
         X_3D = np.hstack((Op[6+i*3:9+i*3], np.array([1.0])))
         point1 = KannalaBrandt_projection((c1aTc1b @ X_3D.T).T, D1, K1)
         point2 = KannalaBrandt_projection((np.linalg.inv(c1Tc2) @ c1aTc1b @ X_3D.T).T, D2, K2)
         point3 = KannalaBrandt_projection(X_3D, D1, K1)
         point4 = KannalaBrandt_projection((np.linalg.inv(c1Tc2) @ X_3D.T).T, D2, K2)

         err1 = xData[0:2,i] - point1[0:2]
         err2 = xData[3:5,i] - point2[0:2]
         err3 = xData[6:8,i] - point3[0:2]
         err4 = xData[9:11,i] - point4[0:2]
                
         res.append(err1[0])
         res.append(err1[1])
         res.append(err2[0])
         res.append(err2[1])
         res.append(err3[0])
         res.append(err3[1])
         res.append(err4[0])
         res.append(err4[1])

    return  np.array(res) 

if __name__ == '__main__':

    img_C1_A = cv.imread('fisheye1_frameA.png')
    img_C1_B = cv.imread('fisheye1_frameB.png')
    img_C2_A = cv.imread('fisheye2_frameA.png')
    img_C2_B = cv.imread('fisheye2_frameB.png')

    x1 = np.loadtxt('x1.txt')
    x2 = np.loadtxt('x2.txt')
    x3 = np.loadtxt('x3.txt')
    x4 = np.loadtxt('x4.txt')

    K1 = np.loadtxt('K_1.txt')
    K2 = np.loadtxt('K_2.txt')

    D1 = np.loadtxt('D1_k_array.txt')
    D2 = np.loadtxt('D2_k_array.txt')

    wTc1 = np.loadtxt('T_wc1.txt')
    wTc2 = np.loadtxt('T_wc2.txt')
    lTr = np.loadtxt('T_leftRight.txt')
    waTwb_gt = np.loadtxt('T_wAwB_gt.txt')
    waTwb_seed = np.loadtxt('T_wAwB_seed.txt')

    #--------------------------------- EXERCISE 1 ---------------------------------#
    v1 = []
    v2 = []
    v3 = []
    v4 = []

    for i in range(x1.shape[1]):
        point1 = KannalaBrandt_unprojection(x1[:, i].T, D1, K1)
        v1.append(point1)
        point2 = KannalaBrandt_unprojection(x2[:, i].T, D2, K2)
        v2.append(point2)
        point3 = KannalaBrandt_unprojection(x3[:, i].T, D1, K1)
        v3.append(point3)
        point4 = KannalaBrandt_unprojection(x4[:, i].T, D2, K2)
        v4.append(point1)

    v1 = np.array(v1).T #(coord, nºpoints)
    v2 = np.array(v2).T
    v3 = np.array(v3).T
    v4 = np.array(v4).T
    

    X1_3D = []

    for i in range(v1.shape[1]):
        X1_3D_i = triangulation(v1[:, i], v2[:, i], lTr) # respect C2_A
        X1_3D.append(X1_3D_i)
    X1_3D = np.array(X1_3D).T

    x1_p = []
    x2_p = []

    for i in range(X1_3D.shape[1]):
        point1 = KannalaBrandt_projection(lTr @ X1_3D[:, i].T, D1, K1)
        x1_p.append(point1)
        point2 = KannalaBrandt_projection(X1_3D[:, i].T, D2, K2)
        x2_p.append(point2)
    
    x1_p = np.array(x1_p).T
    x2_p = np.array(x2_p).T

    plt.figure(2)
    plt.imshow(img_C1_A)
    plt.title('Reprojection 3D points Camera1 A')
    plotResidual(x1, x1_p, 'k-')
    plt.plot(x1[0, :], x1[1, :], 'bo')
    plt.plot(x1_p[0, :], x1_p[1, :], 'rx')
    plotNumberedImagePoints(x1[0:2, :], 'r', 4)
    plt.draw()
    plt.show()

    plt.figure(3)
    plt.imshow(img_C2_A)
    plt.title('Reprojection 3D points Camera2 A')
    plotResidual(x2, x2_p, 'k-')
    plt.plot(x2[0, :], x2[1, :], 'bo')
    plt.plot(x2_p[0, :], x2_p[1, :], 'rx')
    plotNumberedImagePoints(x2[0:2, :], 'r', 4)
    plt.draw()
    plt.show()

    # Camera C1A and C1B
    X_3D_gt = []
    c1aTc1b = np.linalg.inv(wTc1) @ waTwb_gt @ wTc1
    for i in range(v1.shape[1]):
        X_3D_i = triangulation(v1[:, i], v3[:, i], c1aTc1b) # respect to C1 B
        X_3D_gt.append(X_3D_i)
    X_3D_gt = np.array(X_3D_gt).T
   
    x1_p = []
    x3_p = []

    for i in range(X_3D_gt.shape[1]):
        point = KannalaBrandt_projection(c1aTc1b @ X_3D_gt[:, i].T, D1, K1)
        x1_p.append(point)
        point = KannalaBrandt_projection(X_3D_gt[:, i].T, D2, K2)
        x3_p.append(point)

    x1_p = np.array(x1_p).T
    x3_p = np.array(x3_p).T

    plt.figure(5)
    plt.imshow(img_C1_A)
    plt.title('Reprojection 3D points Camera1 A')
    plotResidual(x1, x1_p, 'k-')
    plt.plot(x1[0, :], x1[1, :], 'bo')
    plt.plot(x1_p[0, :], x1_p[1, :], 'rx')
    plotNumberedImagePoints(x1[0:2, :], 'r', 4)
    plt.draw()
    plt.show()

    plt.figure(6)
    plt.imshow(img_C1_B)
    plt.title('Reprojection 3D points Camera1 B')
    plotResidual(x3, x3_p, 'k-')
    plt.plot(x3[0, :], x3[1, :], 'bo')
    plt.plot(x3_p[0, :], x3_p[1, :], 'rx')
    plotNumberedImagePoints(x3[0:2, :], 'r', 4)
    plt.draw()
    plt.show()

    #--------------------------------- EXERCISE 2 ---------------------------------#
    X_3D = []
    c1aTc1b_seed = np.linalg.inv(wTc1) @ waTwb_seed @ wTc1
    for i in range(v1.shape[1]):
        X_3D_i = triangulation(v1[:, i], v3[:, i], c1aTc1b_seed) # respect to C1 B
        X_3D.append(X_3D_i)
    X_3D = np.array(X_3D).T

    waRwb_seed = waTwb_seed[0:3,0:3]
    watwb_seed = waTwb_seed[0:3,3]

    SO_waRwb_seed = crossMatrixInv(linalg.logm(waRwb_seed))

    Op = np.hstack((
        np.hstack((watwb_seed, np.array(SO_waRwb_seed))), np.array(X_3D[0:3, :].T.flatten())
                  ))
    
    xData = np.vstack((x1, x2, x3, x4))
    err = resBundleProjection_n_View(Op, xData, K1, K2, D1, D2, wTc1 ,wTc2, 4, 24)
    print(len(err))

    OpOptim = scOptim.least_squares(resBundleProjection_n_View, Op, args=(xData, K1, K2, D1, D2, wTc1, wTc2, 4, 24,), method='lm')

    points_3D_Op = np.concatenate((OpOptim.x[6: 9], np.array([1.0])), axis=0)
    for i in range(23):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim.x[9+3*i: 12+i*3], np.array([1.0])) ,axis=0)))

    waRwb_Op = linalg.expm(crossMatrix(OpOptim.x[3:6]))
    watwb_Op = OpOptim.x[0:3]
    waTwb_Op = np.vstack((np.concatenate((waRwb_Op, np.expand_dims(watwb_Op, axis=1)), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))

    fig3D = plt.figure(7)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

   
    drawRefSystem(ax, wTc1, '-', 'C1_A')
    drawRefSystem(ax, waTwb_gt @ wTc1, '-', 'C1_B')
    drawRefSystem(ax, waTwb_gt @ wTc2, '-', 'C2_B')
    drawRefSystem(ax, wTc2, '-', 'C2_A')

    drawRefSystem(ax, waTwb_Op @ wTc1, '-', 'C1_B_Op')
    drawRefSystem(ax, waTwb_Op @ wTc2, '-', 'C2_B_Op')
    
    
    points_Op = waTwb_Op @ wTc1 @ points_3D_Op.T

    ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')
    plotNumbered3DPoints(ax, points_Op, 'b', 0.1)

    plt.title('Camera poses')
    plt.show()