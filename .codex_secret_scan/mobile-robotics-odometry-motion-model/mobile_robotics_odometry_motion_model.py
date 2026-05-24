#!/usr/bin/env py_thon3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def wrapToPi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def inverse_motion_model(pose_t_1, pose_t):
    ##STUDENT_CODE:  #TODO Compute rot1,trans and rot2 of the inverse motion model.
    pose_t_1 = np.asarray(pose_t_1, dtype=float)
    pose_t = np.asarray(pose_t, dtype=float)
    dx = pose_t[0] - pose_t_1[0]
    dy = pose_t[1] - pose_t_1[1]
    trans = np.sqrt(dx**2 + dy**2)
    rot1 = wrapToPi(np.arctan2(dy, dx) - pose_t_1[2])
    rot2 = wrapToPi(pose_t[2] - pose_t_1[2] - rot1)

    ##END_STUDENT_CODE:
    return rot1, trans, rot2


def probability_density(mean, variance):
    variance = max(variance,0.00001) #Avoid division by 0!

    ##STUDENT_CODE:  #TODO Compute normal distribution
    density = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(
        -0.5 * (mean**2) / variance
    )

    ##END_STUDENT_CODE
    return density


def motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3=False):

    
    ##STUDENT_CODE:  #TODO Compute p1, p2 and p3!
    rot1, trans, rot2 = inverse_motion_model(x_t_1, x_t)
    rot1_bar, trans_bar, rot2_bar = inverse_motion_model(u_t[0], u_t[1])

    p1 = probability_density(
        wrapToPi(rot1 - rot1_bar),
        alpha[0] * rot1_bar**2 + alpha[1] * trans_bar**2,
    )
    p2 = probability_density(
        trans - trans_bar,
        alpha[2] * trans_bar**2 + alpha[3] * (rot1_bar**2 + rot2_bar**2),
    )
    p3 = probability_density(
        wrapToPi(rot2 - rot2_bar),
        alpha[0] * rot2_bar**2 + alpha[1] * trans_bar**2,
    )

    ##END_STUDENT_CODE 
    if marginalise_p3: 
        return p1 * p2
    else:
        return p1 * p2 * p3


def plot_posterior_belief(x_t_1, u_t, alpha, ret=False, marginalise_p3=False, N_theta = 8):
    size = 150
    gridmap = np.zeros([size, size])
    origin = [np.floor(size/2),np.floor(size/2)]
    res = 0.01

    
    if not marginalise_p3:
        marginalise_p3 = False
        dtheta = 2 * np.pi / N_theta
        

        ##STUDENT_CODE #TODO #3.4 A) Compute gridmap
        theta_values = np.arange(0, 2 * np.pi, dtheta)
        for x_idx in range(size):
            x = (x_idx - origin[0]) * res
            for y_idx in range(size):
                y = (y_idx - origin[1]) * res
                for theta in theta_values:
                    gridmap[y_idx, x_idx] += motion_model(
                        [x, y, theta], x_t_1, u_t, alpha
                    )

        ##END_STUDENT_CODE 

    
    else:
        marginalise_p3 = True
        ##STUDENT_CODE #TODO #3.4 B) Use Marginalised motion_model marginalise_p3=True and pass to motion_model!
        for x_idx in range(size):
            x = (x_idx - origin[0]) * res
            for y_idx in range(size):
                y = (y_idx - origin[1]) * res
                gridmap[y_idx, x_idx] = motion_model(
                    [x, y, 0.0], x_t_1, u_t, alpha, marginalise_p3=True
                )

        ##END_STUDENT_CODE 

    if ret:
        return gridmap
    
    # Normalize
    assert np.sum(gridmap)!=0, "Sum over gridmap is zero! - Error or no implementation!"
    gridmap = gridmap/np.sum(gridmap)
    plt.imshow(1-gridmap, cmap="gray", extent=[-res*(size - origin[0]), res*(size - origin[0]), -res*(size - origin[1]), res*(size - origin[1])])
    plt.show()

def evaluate_sample_odometry(alpha, pose_0, odom, ret=False):
    plot_lims = [(0,6),(2.5,6.5)]

    # get ground truth poses
    gt_poses = [pose_0]
    last_pose = pose_0
    for odom_idx in range(len(odom) - 1):

        ##STUDENT_CODE #TODO: compute xt, yt, theta_t from odometry
        u_t = [odom[odom_idx], odom[odom_idx + 1]]
        rot1, trans, rot2 = inverse_motion_model(*u_t)
        x_t = last_pose[0] + trans * np.cos(last_pose[2] + rot1)
        y_t = last_pose[1] + trans * np.sin(last_pose[2] + rot1)
        theta_t = wrapToPi(last_pose[2] + rot1 + rot2)

        ##END_STUDENT_CODE
        current_pose = [x_t, y_t, theta_t]
        gt_poses.append(current_pose)
        last_pose = current_pose

    gt_poses = np.array(gt_poses)
   
    # draw samples incrementally
    current_pose = pose_0
    estimated_poses = [pose_0]
    samples_for_plot = []

    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_0[0]
    samples[:, 1] = pose_0[1]

    for odom_idx in range(len(odom) - 1):
        # calculate the new samples

        ##STUDENT_CODE # TODO: update all samples and the current_pose
        u_t = [odom[odom_idx], odom[odom_idx + 1]]
        for i in range(num_samples):
            samples[i] = sample_motion_model(samples[i], u_t, alpha)
        current_pose = np.array(
            [
                np.mean(samples[:, 0]),
                np.mean(samples[:, 1]),
                np.arctan2(np.mean(np.sin(samples[:, 2])), np.mean(np.cos(samples[:, 2]))),
            ]
        )

        ##END_STUDENT_CODE
        estimated_poses.append(current_pose)
        samples_for_plot.append(np.copy(samples))


    estimated_poses = np.array(estimated_poses)
    if ret:
        return gt_poses, estimated_poses

    print("estimated_poses: ", estimated_poses)

    #plot all samples, simga elipse and estimated and gt poses
    for idx, samples in enumerate(samples_for_plot):
        sc = plt.scatter(samples[:, 0], samples[:, 1],marker='.')
        color = sc.get_facecolor()[0]
        ax = plt.gca()
        confidence_ellipse(samples[:, 0], samples[:, 1], ax, edgecolor=color,n_std=2)
        plt.scatter(estimated_poses[idx+1, 0], estimated_poses[idx+1 ,1], marker='D',color=color,edgecolor='black')
        plt.scatter(gt_poses[idx+1, 0], gt_poses[idx+1, 1], marker='*',color=color,edgecolor='black', )

    plt.scatter(estimated_poses[0, 0], estimated_poses[0 ,1], marker='D',color='black',edgecolor='black',label="Estimated Pose")
    plt.scatter(gt_poses[0, 0], gt_poses[0, 1], marker='*',color='black',edgecolor='black',label="Ground Truth Pose")    
    plt.plot(0,0,label="2-Sigma Ellipse ~95.45%",color="black",lw=0.5)
        
    plt.xlim(plot_lims[0])
    plt.ylim(plot_lims[1])
    plt.legend()
    plt.show()

def plot_sample_motion_model(pose_t_1, u_t, alpha):
    num_samples = 1000
    samples = np.zeros((num_samples, 3))
    samples[:, 0] = pose_t_1[0]
    samples[:, 1] = pose_t_1[1]

    for i in range(num_samples):
        samples[i] = sample_motion_model(samples[i], u_t, alpha)

    plt.figure()
    plt.plot(samples[:,0], samples[:,1], '.')
    plt.axis('equal')
    plt.show()

def get_sample(std): 
    #Irwin–Hall -> Using Central Limit Theorem to generate cheap normal distributed samples. 
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-std,std)
    
    return 0.5*tot


def sample_motion_model(pose_t_1, u_t, alpha):
    ##STUDENT_CODE #TODO Compute x_t, y_t and theta_t
    pose_t_1 = np.asarray(pose_t_1, dtype=float)
    rot1, trans, rot2 = inverse_motion_model(u_t[0], u_t[1])

    rot1_hat = rot1 - get_sample(np.sqrt(alpha[0] * rot1**2 + alpha[1] * trans**2))
    trans_hat = trans - get_sample(
        np.sqrt(alpha[2] * trans**2 + alpha[3] * (rot1**2 + rot2**2))
    )
    rot2_hat = rot2 - get_sample(np.sqrt(alpha[0] * rot2**2 + alpha[1] * trans**2))

    x_t = pose_t_1[0] + trans_hat * np.cos(pose_t_1[2] + rot1_hat)
    y_t = pose_t_1[1] + trans_hat * np.sin(pose_t_1[2] + rot1_hat)
    theta_t = wrapToPi(pose_t_1[2] + rot1_hat + rot2_hat)

    ##END_STUDENT_CODE
    return x_t, y_t, theta_t



from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
#https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
