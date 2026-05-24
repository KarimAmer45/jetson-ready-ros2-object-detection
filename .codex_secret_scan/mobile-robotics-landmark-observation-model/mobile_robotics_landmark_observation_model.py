#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def dist(x, y):
    return np.sqrt(((x[0]-y[0])**2)+((x[1]-y[1])**2))

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y

def landmark_observation_model(z,sigma_r, b, x):
    ##STUDENT_CODE:  #TODO
    expected_range = dist(b, x)
    prob = normpdf(z, expected_range, sigma_r)

    ##END_STUDENT_CODE:
    return prob

def observation_likelihood(z,sigma_z, b, gridmap):
    ##STUDENT_CODE:  #TODO
    likelihood = np.zeros_like(gridmap, dtype=float)
    for i in range(gridmap.shape[0]):
        for j in range(gridmap.shape[1]):
            likelihood[i, j] = landmark_observation_model(
                z, sigma_z, b, np.array([i, j])
            )

    ##END_STUDENT_CODE:
    return likelihood

def joint_observation_likelihood(z, sigma_z, b, gridmap):
    ##STUDENT_CODE:  #TODO    
    z = np.atleast_1d(z)
    sigma_z = np.atleast_1d(sigma_z)
    b = np.atleast_2d(b)

    joint_likelihood = np.ones_like(gridmap, dtype=float)
    for z_i, sigma_i, beacon_i in zip(z, sigma_z, b):
        joint_likelihood *= observation_likelihood(z_i, sigma_i, beacon_i, gridmap)

    ##END_STUDENT_CODE:
    return joint_likelihood



def plot_setup(num_obs=1):
    plt.close()
    fig, axes = plt.subplots(1, num_obs, figsize=(10, 5))
    if num_obs == 1:
        ytitle = 1.05
        ysubtitle = 0.95
    else:
        ytitle = 0.88
        ysubtitle = 0.8

    fig.suptitle("Likelihood Maps of Beacon Observations", fontsize=16, fontweight='bold',y=ytitle)
    fig.text(0.5, ysubtitle, "Solid line = observed distance, dotted lines = ±2σ uncertainty",
            ha='center', fontsize=10)

    return axes

def plot_helper(likelihood,colors,z_all,sigma_z_all, beacon_all, num_obs,ax,n_sigma = 2):

    #If single observation not passed as array
    if isinstance(colors,str):
        colors = [colors]
        z_all = np.array([z_all])
        sigma_z_all = np.array([sigma_z_all])
        beacon_all = np.array([beacon_all])
    else:
        colors = np.tile(colors, int(np.ceil(num_obs/len(colors))))[:num_obs] #Ensures enough colors are defined - wraps around color list

    ax.scatter(beacon_all[:,0],beacon_all[:,1], s=100,c=colors)

    for j in range(num_obs):

        obs_color = colors[j]
        circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=z_all[j], edgecolor=obs_color, facecolor='none', lw=2)
        ax.add_patch(circle)

        # ±2σ (n_sigma) uncertainty
        inner_radius = max(z_all[j] - n_sigma*sigma_z_all[j],0)
        outer_radius = z_all[j] + n_sigma*sigma_z_all[j]
        
        uncertainty_circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=inner_radius,
                                    edgecolor=obs_color, facecolor='none',ls='--',lw=0.5)
        ax.add_patch(uncertainty_circle)
        uncertainty_circle = Circle((beacon_all[j,0], beacon_all[j,1]), radius=outer_radius,
                                    edgecolor=obs_color, facecolor='none',ls='--',lw=0.5)
        ax.add_patch(uncertainty_circle)

    #Needs to be transposed to match other plots
    ax.matshow(np.transpose(likelihood), origin="lower", cmap='gray')
