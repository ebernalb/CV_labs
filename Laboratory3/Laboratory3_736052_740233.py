
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import scipy.optimize as scOptim

import cv2

def car2spherical_unity(x, y, z):
    xy = x**2 + y**2
    if z == 0:
        theta = np.pi / 2.0
    else:
        theta = np.arctan2(np.sqrt(xy), z)
    if x == 0:
        phi = (np.pi / 2.0) * np.sign(y)
    else:
        phi = np.arctan2(y, x)
    return np.array([theta, phi])

def spherical_unity2car(theta, phi):

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

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

def crossMatrixInv(M):
     x = [M[2, 1], M[0, 2], M[1, 0]]
     return x

def crossMatrix(x):
     M = np.array([[0, -x[2], x[1]],
     [x[2], 0, -x[0]],
     [-x[1], x[0], 0]], dtype="object")
     return M

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

def calculate3DPoints(points1,points2,Pc1,Pc2):
    List_P3D = []

    for i in range(points1.shape[1]):
        A = []
        A.append(np.array([
            Pc1[2, 0] * points1[0, i] - Pc1[0, 0],
            Pc1[2, 1] * points1[0, i] - Pc1[0, 1],
            Pc1[2, 2] * points1[0, i] - Pc1[0, 2],
            Pc1[2, 3] * points1[0, i] - Pc1[0, 3]
        ]))
        A.append(np.array([
            Pc1[2, 0] * points1[1, i] - Pc1[1, 0],
            Pc1[2, 1] * points1[1, i] - Pc1[1, 1],
            Pc1[2, 2] * points1[1, i] - Pc1[1, 2],
            Pc1[2, 3] * points1[1, i] - Pc1[1, 3]
        ]))
        A.append(np.array([
            Pc2[2, 0] * points2[0, i] - Pc2[0, 0],
            Pc2[2, 1] * points2[0, i] - Pc2[0, 1],
            Pc2[2, 2] * points2[0, i] - Pc2[0, 2],
            Pc2[2, 3] * points2[0, i] - Pc2[0, 3]
        ]))
        A.append(np.array([
            Pc2[2, 0] * points2[1, i] - Pc2[1, 0],
            Pc2[2, 1] * points2[1, i] - Pc2[1, 1],
            Pc2[2, 2] * points2[1, i] - Pc2[1, 2],
            Pc2[2, 3] * points2[1, i] - Pc2[1, 3]
        ]))

        A = np.array(A)
        _, S, V = np.linalg.svd(A, full_matrices=True)

        Point_3D = V.T[:, -1]
        Point_3D = Point_3D / Point_3D[3]

        List_P3D.append(Point_3D)
    List_P3D = np.array(List_P3D)
    return List_P3D.T

def n_Points_in_Front(x_point, C1_T_C2 ): #C2_T_C1
    votos = 0

    for x in x_point.T:

        x_prj = C1_T_C2 @ x #C2_T_C1
        if x[2] >= 0 and x_prj[2] >= 0:
            votos += 1
    return votos

def resBundleProjection_n_View(Op, xData, nCameras, K_c, nPoints):
    '''
    Op[0:2] -> C1: theta, phi
    Op[2:5] -> C1: Rx,Ry,Rz
    Op[5:8] -> C2: tx,ty,tz
    Op[8:11] -> C2: Rx,Ry,Rz
    ...
    Op[8:nPoints*3+6] -> 3DXx,3DXy,3DXz
    '''

    P_can = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    theta_ext_1 = K_c @ P_can

    R = linalg.expm(crossMatrix(Op[2:5]))
    t = spherical_unity2car(Op[0], Op[1])
    theta_ext_2 = K_c @ np.hstack((R, np.expand_dims(t, axis=1)))

    theta_ext=[]
    theta_ext.append(theta_ext_1)
    theta_ext.append(theta_ext_2)
    for i in range(nCameras-2):
        R = linalg.expm(crossMatrix(Op[8+6*i:11+6*i]))
        theta_ext_c = K_c @ np.hstack((R, np.expand_dims(Op[5+6*i:8+6*i], axis=1)))
        theta_ext.append(theta_ext_c)

    Xpoints = []
    for i in range(nCameras):
        x = xData[i*3:2+i*3, :] / xData[2+3*i, :]
        Xpoints.append(x)

    res = []
    for i in range(nCameras):
        for j in range(nPoints):

            X_3D = np.hstack((Op[6*(nCameras-1)+j*3-1: 6*(nCameras-1)+j*3+2], np.array([1.0])))
            projection = theta_ext[i] @ X_3D
            projection = projection[0:2] / projection[2]
            err = Xpoints[i][:,j] - projection

            res.append(err[0])
            res.append(err[1])
    return np.array(res)
def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
    '''
    Op[0:2] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[5:5 + nPoints*3] -> 3DXx,3DXy,3DXz
    '''

    P_can = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    R = linalg.expm(crossMatrix(Op[2:5]))
    t = spherical_unity2car(Op[0], Op[1])
    theta_ext_2 = K_c @ np.hstack((R, np.expand_dims(t, axis=1))) #Proyection matrix
    theta_ext_1 = K_c @ P_can
    res = []
    for i in range(nPoints):

        X_3D = np.hstack((Op[5+i*3: 5+i*3+3], np.array([1.0])))
        projection1 = theta_ext_1 @ X_3D
        projection1 = projection1[0:2] / projection1[2]
        x1 = x1Data[0:2, i] / x1Data[2, i]
        res1 = x1 - projection1

        projection2 = theta_ext_2 @ X_3D
        projection2 = projection2[0:2] / projection2[2]
        x2 = x2Data[0:2, i] / x2Data[2, i]
        res2 = x2 - projection2

        res.append(res1[0])
        res.append(res1[1])
        res.append(res2[0])
        res.append(res2[1])

    return np.array(res)

if __name__ == '__main__':

    # Read the images
    timestamp2 = '1403715282262142976'
    timestamp1 = '1403715413262142976'
    timestamp3 = '1403715571462142976'
    path_image_1 = timestamp1 + '_undistort.png'
    path_image_2 = timestamp2 + '_undistort.png'
    path_image_3 = timestamp3 + '_undistort.png'
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)
    image_pers_3 = cv2.imread(path_image_3)

    K = np.loadtxt('K_c.txt')
    c1_F_c2 = np.loadtxt('F_21.txt')
    x1 = np.loadtxt('x1.txt')
    x2 = np.loadtxt('x2.txt')
    x3 = np.loadtxt('x3.txt')
    wTc1 = np.loadtxt('T_wc1.txt')
    wTc2 = np.loadtxt('T_wc2.txt')
    wTc3 = np.loadtxt('T_wc3.txt')
    P_can = np.eye(3, 4)
    P_C1_True = K @ P_can @ wTc1
    P_C2_True = K @ P_can @ wTc2
    c2Tc1_True = np.linalg.inv(wTc2) @ wTc1
    c3Tc1_True = np.linalg.inv(wTc3) @ wTc1

    image_1 = cv2.imread('1403715282262142976_undistort.png')
    image_2 = cv2.imread('1403715413262142976_undistort.png')

    ############################## EXERCISE 2.1 Pose estimation ########################################################
    E = K.T @ c1_F_c2.T @ K

    U, S, V = np.linalg.svd(E)
    t = U[:, -1]
    W = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    R_90_pos = U @ W @ V
    R_90_neg = U @ W.T @ V
    I = np.concatenate((np.identity(3), np.array([[0.0, 0.0, 0.0]]).T), axis=1)

    P1_est = K @ I

    P2_a_est = K @ np.concatenate((R_90_pos, np.array([[t[0]], [t[1]],[t[2]] ])), axis=1)
    P2_b_est = K @ np.concatenate((R_90_pos, np.array([[-t[0]], [-t[1]], [-t[2]]])), axis=1)
    P2_c_est = K @ np.concatenate((R_90_neg, np.array([[t[0]], [t[1]], [t[2]]])), axis=1)
    P2_d_est = K @ np.concatenate((R_90_neg, np.array([[-t[0]], [-t[1]], [-t[2]]])), axis=1)

    T_c2_c1_a = np.vstack((np.concatenate((R_90_pos, np.array([[t[0]], [t[1]],[t[2]] ])), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))
    T_c2_c1_b = np.vstack((np.concatenate((R_90_pos, np.array([[-t[0]], [-t[1]], [-t[2]]])), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))
    T_c2_c1_c = np.vstack((np.concatenate((R_90_neg, np.array([[t[0]], [t[1]], [t[2]]])), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))
    T_c2_c1_d = np.vstack((np.concatenate((R_90_neg, np.array([[-t[0]], [-t[1]], [-t[2]]])), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))

    points_P2a = calculate3DPoints(x1, x2, P1_est, P2_a_est)
    points_P2b = calculate3DPoints(x1, x2, P1_est, P2_b_est)
    points_P2c = calculate3DPoints(x1, x2, P1_est, P2_c_est)
    points_P2d = calculate3DPoints(x1, x2, P1_est, P2_d_est)

    n_votos_a = n_Points_in_Front(points_P2a, T_c2_c1_a)
    n_votos_b = n_Points_in_Front(points_P2b, T_c2_c1_b)
    n_votos_c = n_Points_in_Front(points_P2c, T_c2_c1_c)
    n_votos_d = n_Points_in_Front(points_P2d, T_c2_c1_d)

    print('nº votos A: ', n_votos_a)
    print('nº votos B: ', n_votos_b)
    print('nº votos C: ', n_votos_c)
    print('nº votos D: ', n_votos_d)

    ### Plot 3D points Pose estimation
    fig3D = plt.figure(1)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, wTc1, '-', 'C1_True')
    drawRefSystem(ax, wTc1 @ np.linalg.inv(T_c2_c1_d), '-', 'C2_PE')
    drawRefSystem(ax, wTc2, '-', 'C2_True')

    points_est = wTc1 @ points_P2d

    ax.scatter(points_est[0, :]/points_est[3, :], points_est[1, :]/points_est[3, :], points_est[2, :]/points_est[3, :], marker='.')
    plotNumbered3DPoints(ax, points_est, 'b', 0.1)

    P3D_Scene = np.loadtxt('X_w.txt')
    ax.scatter(P3D_Scene[0, :], P3D_Scene[1, :], P3D_Scene[2, :], marker='.')
    plotNumbered3DPoints(ax, P3D_Scene, 'r', 0.1)

    plt.title('3D points Pose estimation (red=True data)')
    plt.show()

    ### Draw residuals

    x1_p = P1_est @ points_P2d
    x2_p = P2_d_est @ points_P2d
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]

    plt.figure(24)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals before Bundle adjustment Image1')
    plotResidual(x1, x1_p, 'k-')
    plt.plot(x1[0, :], x1[1, :], 'bo')
    plt.plot(x1_p[0, :], x1_p[1, :], 'rx')
    plotNumberedImagePoints(x1[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    plt.figure(25)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals before Bundle adjustment Image2')
    plotResidual(x2, x2_p, 'k-')
    plt.plot(x2[0, :], x2[1, :], 'bo')
    plt.plot(x2_p[0, :], x2_p[1, :], 'rx')
    plotNumberedImagePoints(x2[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    #################### EXERCISE 2.2 Bundle adjustment ##############################################################

    P2_est = P2_d_est
    estimated_points3D = calculate3DPoints(x1, x2, P1_est, P2_est)
    c2Tc1_est = T_c2_c1_d

    SO_c2Rc1_est = crossMatrixInv(linalg.logm(c2Tc1_est[0:3, 0:3]))
    Spherical_c2tc1 = car2spherical_unity(c2Tc1_est[0, 3], c2Tc1_est[1, 3], c2Tc1_est[2, 3])
    points3D = estimated_points3D[0:3, :]
    Op = np.hstack((
        np.hstack((Spherical_c2tc1, np.array(SO_c2Rc1_est))), np.array(points3D.T.flatten())
                  ))

    err = resBundleProjection(Op, x1, x2, K, 35)
    print(len(err))

    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(x1, x2, K, 35,), method='lm')

    points_3D_Op = np.concatenate((OpOptim.x[5: 8], np.array([1.0])), axis=0)

    for i in range(34):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim.x[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))

    c2Rc1_Op = linalg.expm(crossMatrix(OpOptim.x[2:5]))
    c2tc1_Op = spherical_unity2car(OpOptim.x[0], OpOptim.x[1])
    P2_Op = K @ np.concatenate((c2Rc1_Op, np.expand_dims(c2tc1_Op, axis=1)), axis=1)
    c2Tc1_Op = np.vstack((np.concatenate((c2Rc1_Op, np.expand_dims(c2tc1_Op, axis=1)), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))


    #### Draw 3D ################
    fig3D = plt.figure(2)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, wTc1, '-', 'C1')
    #drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA')
    drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA_scaled')
    drawRefSystem(ax, wTc2, '-', 'C2_True')

    points_Op = wTc1 @ (points_3D_Op).T

    ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')
    plotNumbered3DPoints(ax, points_Op, 'b', 0.1)

    ax.scatter(P3D_Scene[0, :], P3D_Scene[1, :], P3D_Scene[2, :], marker='.')
    plotNumbered3DPoints(ax, P3D_Scene, 'r', 0.1)

    plt.title('3D points Bundle adjustment (red=True data)')
    plt.show()

    #### Plot residual bundel adj ##############

    x1_p = P1_est @ points_3D_Op.T
    x2_p = P2_Op @ points_3D_Op.T
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]

    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals after Bundle adjustment Image1')
    plotResidual(x1, x1_p, 'k-')
    plt.plot(x1[0, :], x1[1, :], 'bo')
    plt.plot(x1_p[0, :], x1_p[1, :], 'rx')
    plotNumberedImagePoints(x1[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals after Bundle adjustment Image2')
    plotResidual(x2, x2_p, 'k-')
    plt.plot(x2[0, :], x2[1, :], 'bo')
    plt.plot(x2_p[0, :], x2_p[1, :], 'rx')
    plotNumberedImagePoints(x2[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    ###################### EXERCISE 3 P-N-P ###########################################################################

    Points_3D_Scene = np.float32(points_3D_Op[:, 0:3])
    Points_C3 = np.ascontiguousarray(x3[0:2, :].T).reshape((x3.shape[1], 1, 2))
    Coeff=[]
    retval, rvec, tvec = cv2.solvePnP(objectPoints=Points_3D_Scene, imagePoints=Points_C3, cameraMatrix=K,
                                      distCoeffs=np.array(Coeff), flags=cv2.SOLVEPNP_EPNP)


    c3Rc1_PnP = linalg.expm(crossMatrix(rvec))
    c3tc1_PnP = tvec
    c3Tc1_PnP = np.vstack((np.concatenate((c3Rc1_PnP, c3tc1_PnP), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))

    fig3D = plt.figure(8)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, wTc1, '-', 'C1_True')
    drawRefSystem(ax, wTc3, '-', 'C3_True')
    drawRefSystem(ax, wTc1 @ np.linalg.inv(c3Tc1_PnP), '-', 'C3_PnP')
    plt.title('3D camera poses PnP')
    plt.draw()
    plt.show()

    ####################### EXERCISE 4 Bundle adjustment nCameras ######################################################

    SO_c3Rc1_PnP = rvec
    Op = np.hstack((
        np.hstack((Spherical_c2tc1, np.array(SO_c2Rc1_est))), np.hstack((c3tc1_PnP[:, 0], SO_c3Rc1_PnP[:, 0])), np.array(points3D.T.flatten())
                  ))

    xData = np.vstack((np.vstack((x1, x2)), x3))
    err = resBundleProjection_n_View(Op, xData, 3, K, 35)
    print(len(err))

    OpOptim = scOptim.least_squares(resBundleProjection_n_View, Op, args=(xData, 3, K, 35,), method='lm')

    points_3D_Opn = np.concatenate((OpOptim.x[11: 14], np.array([1.0])), axis=0)

    #### Scale ##################

    c2tc1_True = np.array([[c2Tc1_True[0, 3], c2Tc1_True[1, 3], c2Tc1_True[2, 3]]])

    scale = np.linalg.norm(c2tc1_True)
    print('SCALE: ', scale)

    for i in range(34):
        points_3D_Opn = np.vstack(
            (points_3D_Opn, np.concatenate((OpOptim.x[14 + 3 * i: 14 + 3 * i + 3], np.array([1.0])), axis=0)))

    c2Rc1_Opn = linalg.expm(crossMatrix(OpOptim.x[2:5]))
    c2tc1_Opn = spherical_unity2car(OpOptim.x[0], OpOptim.x[1]) * scale
    P2_Opn = K @ np.concatenate((c2Rc1_Opn, np.expand_dims(c2tc1_Opn, axis=1)), axis=1)
    c2Tc1_Opn = np.vstack(
        (np.concatenate((c2Rc1_Opn, np.expand_dims(c2tc1_Opn, axis=1)), axis=1), np.array([0.0, 0.0, 0.0, 1.0])))

    c3Rc1_Opn = linalg.expm(crossMatrix(OpOptim.x[8:11]))
    c3tc1_Opn = OpOptim.x[5:8] * scale
    P3_Opn = K @ np.concatenate((c3Rc1_Opn, np.array([[c3tc1_Opn[0]], [c3tc1_Opn[1]], [c3tc1_Opn[2]]])), axis=1)
    c3Tc1_Opn = np.vstack(
        (np.concatenate((c3Rc1_Opn, np.array([[c3tc1_Opn[0]], [c3tc1_Opn[1]], [c3tc1_Opn[2]]])), axis=1),
         np.array([0.0, 0.0, 0.0, 1.0])))

    #### Draw 3D
    fig3D = plt.figure(32)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Opn), '-', 'C2_BA')
    drawRefSystem(ax, wTc1 @ np.linalg.inv(c3Tc1_Opn), '-', 'C3_BA')
    drawRefSystem(ax, wTc2, '-', 'C2_True')
    drawRefSystem(ax, wTc3, '-', 'C3_True')

    points_Opn = wTc1 @ (points_3D_Opn * scale).T

    ax.scatter(points_Opn[0, :], points_Opn[1, :], points_Opn[2, :], marker='.')
    plotNumbered3DPoints(ax, points_Opn, 'b', 0.1)

    ax.scatter(P3D_Scene[0, :], P3D_Scene[1, :], P3D_Scene[2, :], marker='.')
    plotNumbered3DPoints(ax, P3D_Scene, 'r', 0.1)
    plt.title('3D points Bundle adjustment 3 cameras(red=True data)')
    plt.show()

    # Plot residual bundle adj nC

    x1_p = P1_est @ points_3D_Opn.T
    x2_p = P2_Opn @ points_3D_Opn.T
    x3_p = P3_Opn @ points_3D_Opn.T
    x1_p /= x1_p[2, :]
    x2_p /= x2_p[2, :]
    x3_p /= x3_p[2, :]

    plt.figure(40)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals Bundle adjustment 3 cameras Image1')
    plotResidual(x1, x1_p, 'k-')
    plt.plot(x1[0, :], x1[1, :], 'bo')
    plt.plot(x1_p[0, :], x1_p[1, :], 'rx')
    plotNumberedImagePoints(x1[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    plt.figure(50)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals Bundle adjustment 3 cameras Image2')
    plotResidual(x2, x2_p, 'k-')
    plt.plot(x2[0, :], x2[1, :], 'bo')
    plt.plot(x2_p[0, :], x2_p[1, :], 'rx')
    plotNumberedImagePoints(x2[0:2, :], 'r', 4)
    plt.draw()

    plt.show()

    plt.figure(60)
    plt.imshow(image_pers_3, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals Bundle adjustment 3 cameras Image3')
    plotResidual(x3, x3_p, 'k-')
    plt.plot(x3[0, :], x3[1, :], 'bo')
    plt.plot(x3_p[0, :], x3_p[1, :], 'rx')
    plotNumberedImagePoints(x3[0:2, :], 'r', 4)
    plt.draw()

    print('2D projections are worsened by the scale compensation but it marginally improves the predicted c2 position in 3D')
    plt.show()
