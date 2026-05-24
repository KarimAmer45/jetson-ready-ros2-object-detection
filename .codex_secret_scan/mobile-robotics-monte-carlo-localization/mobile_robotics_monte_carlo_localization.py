#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter
from IPython.display import Image, display, clear_output
import matplotlib
import numpy as np
import os
import pickle
import timeit
import inspect

POINTSIZE_TRESHHOLDS = [(50, 10), (500, 2), (5000, 0.5)]
TODO = None


def _wrap_to_pi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def sample_motion_model_odometry(pose_t_1, u_t, alpha):
    rot1, trans, rot2 = u_t

    ##STUDENT_CODE #TODO Q1: Compute x_t, y_t and theta_t

    rot1_hat = rot1 - get_sample(np.sqrt(alpha[0] * abs(rot1) + alpha[1] * abs(trans)))
    trans_hat = trans - get_sample(
        np.sqrt(alpha[2] * abs(trans) + alpha[3] * (abs(rot1) + abs(rot2)))
    )
    rot2_hat = rot2 - get_sample(np.sqrt(alpha[0] * abs(rot2) + alpha[1] * abs(trans)))

    # For this task (compared to EX03) you have been already given the rot1, trans and rot2 components.

    # Please follow the notation of the lecture -> The variance is a sum of absolute values not squares!

    x_t = pose_t_1[0] + trans_hat * np.cos(pose_t_1[2] + rot1_hat)
    y_t = pose_t_1[1] + trans_hat * np.sin(pose_t_1[2] + rot1_hat)
    theta_t = float(_wrap_to_pi(pose_t_1[2] + rot1_hat + rot2_hat))


    ##END_STUDENT_CODE
    shape_check("x_t",x_t,())
    shape_check("y_t",y_t,())
    shape_check("theta_t",theta_t,())
    return np.array([x_t, y_t, theta_t])


def compute_weights(
    x_pose, z_obs, gridmap, likelihood_map, map_res, p_outside=0.1, max_range=10
):
    ##STUDENT_CODE: #TODO Q2: Compute weights for each sample: The observation model assumes beams are conditionally independent!

    ranges = z_obs[1, :]
    angles = z_obs[0, :]

    # Filter sensor readings that do not lie within max_range
    valid = (ranges > 0) & (ranges < max_range)
    ranges = ranges[valid]
    angles = angles[valid]
    if ranges.size == 0:
        weight = float(p_outside)
        ##END_STUDENT_CODE
        shape_check("weight",weight,())
        return weight

    # Transform observation to gridmap coordinates

    z_in_map = ranges2cells(ranges, angles, x_pose, gridmap, map_res)

    # Count beams that land outside the map
    x_idx = z_in_map[0, :]
    y_idx = z_in_map[1, :]
    rows, cols = likelihood_map.shape
    inside = (x_idx >= 0) & (x_idx < cols) & (y_idx >= 0) & (y_idx < rows)
    outside_count = np.count_nonzero(~inside)

    # Compute probabilty for those outside beam occurences.

    # Give all of them the same weight of p_outside

    # Obtain probabilties from beams that end (i,j) within the gridmap

    p_ij = likelihood_map[y_idx[inside], x_idx[inside]]

    # Compute final weight by combining both subsets of beams.
    p_ij = np.maximum(p_ij, 1e-12)
    weight = float((p_outside ** outside_count) * np.prod(p_ij))


    ##END_STUDENT_CODE
    shape_check("weight",weight,())
    return weight


def low_variance_resampler(particles, weights):
    ##STUDENT_CODE: #TODO Q3: Implement the low variance resampling strategy

    N = particles.shape[0]
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    resampled_particles = np.zeros_like(particles)
    resampled_idx = np.zeros(N, dtype=int)
    cumulative_weights = np.cumsum(weights)

    # Compute random resample_offset in range [0:1/N)

    resample_offset = float(np.random.uniform(0, 1 / N))

    # Iterate over N evenenly distributed bins [0,1) shifted by offset_start

    i = 0
    for j in range(N):

        # Find first index of cummulative weights that overlay the left bin border
        position = resample_offset + j / N
        while position > cumulative_weights[i] and i < N - 1:
            i += 1

        # Store resampled particle and corresponding index

        resampled_particles[j] = particles[i]

        resampled_idx[j] = i




    ##END_STUDENT_CODE
    assert resampled_particles[:, 0].size == resampled_idx.size, (
        "Both arrays need to have the same number of particles!"
    )
    shape_check("resampled_particles",resampled_particles,(N,3))
    shape_check("resampled_idx",resampled_idx,(N,))
    shape_check("resample_offset",resample_offset,float)
    return resampled_particles, resampled_idx, resample_offset


def mc_localization(
    odom,
    z,
    num_particles,
    particles,
    noise,
    gridmap,
    likelihood_map,
    map_res,
    img_map,
    viz_timestep_intervall=None,
    parallel_mode=False,
    surpress_pb=False
):
    # Init GifWriter
    file_name, fig, GifWriter = init_plot_mc_localization(num_particles, parallel_mode)
    init_particles = np.copy(particles)
    VECTORIZED = parallel_mode

    with GifWriter.saving(fig, file_name, dpi=150):
        ##STUDENT_CODE: #TODO Q4: Implement the complete mc_localization

        # Iterate over all timesteps

        for timestep in range(len(odom)):

            if not VECTORIZED:

                ##STUDENT_CODE: #TODO Q4:

                weights = np.zeros(num_particles)

                # Iterate over all particles
                for particle_idx in range(num_particles):

                    # Compute forward motion model
                    particles[particle_idx] = sample_motion_model_odometry(
                        particles[particle_idx], odom[timestep], noise
                    )

                    # Compute weight for particle
                    weights[particle_idx] = compute_weights(
                        particles[particle_idx],
                        z[timestep],
                        gridmap,
                        likelihood_map,
                        map_res,
                    )

            else:

                ##STUDENT_CODE #TODO Q5

                particles = sample_motion_model_odometry_parallel(
                    particles, odom[timestep], noise
                )
                weights = compute_weights_parallel(
                    particles, z[timestep], gridmap, likelihood_map, map_res
                )

            # Normalise the weights across all particles
            weight_sum = np.sum(weights)
            if weight_sum <= 0 or not np.isfinite(weight_sum):
                weights = np.ones(num_particles) / num_particles
            else:
                weights = weights / weight_sum

            if not VECTORIZED:

                # Apply low variance resampling

                particles, _, _ = low_variance_resampler(particles, weights)

            else:

                ##STUDENT_CODE #TODO Q5: do not loop over bins!

                particles, _, _ = low_variance_resampler_parallel(particles, weights)




            ##END_STUDENT_CODE
            
            if not surpress_pb:
                progress_bar(timestep+1,len(odom))

            # Intervall plotting result
            if (
                viz_timestep_intervall is not None
                and timestep % viz_timestep_intervall == 0
            ):
                # print(f"Adding timestep {timestep} to gif")
                plot_mc_localization(
                    GifWriter,
                    init_particles,
                    particles,
                    img_map,
                    map_res,
                    f"Timestep: {timestep}",
                    "Runtime particles",
                    ".b",
                )
        if not surpress_pb:
            clear_output(wait=True)

        # Final plotting result
        if viz_timestep_intervall is not None:
            plot_mc_localization(
                GifWriter,
                init_particles,
                particles,
                img_map,
                map_res,
                f"Final timestep: {timestep}",
                "Final particles",
                ".g",
            )
            file_name2 = "FINAL_STATE_" + file_name.replace("gif", "png")
            print(f"Saved GIF: {file_name}\n      IMG: {file_name2}")
            plt.savefig(file_name2)
        plt.close("all")
    if viz_timestep_intervall is not None:
        display(Image(filename=file_name))
    else:
        return particles

def ranges2cells_parallel(r_ranges, r_angles, w_poses, gridmap, map_res):
    ##STUDENT_CODE #TODO Q5

    r_points = ranges2points_parallel(r_ranges, r_angles)
    w_T_r = v2t_parallel(w_poses)
    w_points = np.einsum("bij,jk->bik", w_T_r, r_points)
    m_points = world2map_parallel(w_points, gridmap, map_res)[:, 0:2, :]

    ##END_STUDENT_CODE
    shape_check("m_points",m_points,(w_poses.shape[0],2,r_ranges.shape[0]))
    return m_points

def ranges2points_parallel(ranges, angles):
    ##STUDENT_CODE #TODO Q5

    points_hom = np.vstack(
        [
            ranges * np.cos(angles),
            ranges * np.sin(angles),
            np.ones_like(ranges),
        ]
    )

    ##END_STUDENT_CODE
    shape_check("points_hom",points_hom,(3,ranges.shape[0]))
    return points_hom


def v2t_parallel(poses):
    ##STUDENT_CODE #TODO Q5

    tr = np.zeros((poses.shape[0], 3, 3))
    c = np.cos(poses[:, 2])
    s = np.sin(poses[:, 2])
    tr[:, 0, 0] = c
    tr[:, 0, 1] = -s
    tr[:, 0, 2] = poses[:, 0]
    tr[:, 1, 0] = s
    tr[:, 1, 1] = c
    tr[:, 1, 2] = poses[:, 1]
    tr[:, 2, 2] = 1

    ##END_STUDENT_CODE
    shape_check("tr",tr,(poses.shape[0],3,3))
    return tr

def world2map_parallel(poses, gridmap, map_res):
    ##STUDENT_CODE #TODO Q5

    max_y = np.size(gridmap, 0) - 1
    new_poses = np.zeros_like(poses)
    if poses.ndim == 3:
        new_poses[:, 0, :] = np.round(poses[:, 0, :] / map_res)
        new_poses[:, 1, :] = max_y - np.round(poses[:, 1, :] / map_res)
        new_poses[:, 2, :] = poses[:, 2, :]
    else:
        new_poses[:, 0] = np.round(poses[:, 0] / map_res)
        new_poses[:, 1] = max_y - np.round(poses[:, 1] / map_res)
        new_poses[:, 2] = poses[:, 2]

    ##END_STUDENT_CODE
    shape_check("new_poses",new_poses,poses.shape)
    return new_poses.astype(int)

def sample_motion_model_odometry_parallel(poses_t_1, u_t, alpha):
    ##STUDENT_CODE #TODO Q5

    rot1, trans, rot2 = u_t
    count = poses_t_1.shape[0]
    rot1_hat = rot1 - get_sample_parallel(
        np.sqrt(alpha[0] * abs(rot1) + alpha[1] * abs(trans)), count
    )
    trans_hat = trans - get_sample_parallel(
        np.sqrt(alpha[2] * abs(trans) + alpha[3] * (abs(rot1) + abs(rot2))), count
    )
    rot2_hat = rot2 - get_sample_parallel(
        np.sqrt(alpha[0] * abs(rot2) + alpha[1] * abs(trans)), count
    )

    x_t = poses_t_1[:, 0] + trans_hat * np.cos(poses_t_1[:, 2] + rot1_hat)
    y_t = poses_t_1[:, 1] + trans_hat * np.sin(poses_t_1[:, 2] + rot1_hat)
    theta_t = _wrap_to_pi(poses_t_1[:, 2] + rot1_hat + rot2_hat)

    ##END_STUDENT_CODE
    shape_check("x_t",x_t,(poses_t_1.shape[0],))
    shape_check("y_t",y_t,(poses_t_1.shape[0],))
    shape_check("theta_t",theta_t,(poses_t_1.shape[0],))
    return np.stack([x_t, y_t, theta_t], axis=1)


def get_sample_parallel(std, count):
    ##STUDENT_CODE #TODO Q5: 

    tot = np.zeros(count)
    for _ in range(12):
        tot += np.random.uniform(-std, std, size=count)

    ##END_STUDENT_CODE
    shape_check("tot",tot,(count,))
    return 0.5 * tot

def compute_weights_parallel(
    x_poses, z_obs, gridmap, likelihood_map, map_res, p_outside=0.1, max_range=10
):
    ##STUDENT_CODE #TODO Q5: Compute particle weights

    B = x_poses.shape[0]
    ranges = z_obs[1, :]
    angles = z_obs[0, :]

    # Filter sensor readings that do not lie within max_range
    valid = (ranges > 0) & (ranges < max_range)
    ranges = ranges[valid]
    angles = angles[valid]
    if ranges.size == 0:
        weights = np.full(B, p_outside, dtype=float)
        ##END_STUDENT_CODE
        shape_check("weights",weights,(B,))
        return weights
    z_in_map = ranges2cells_parallel(ranges, angles, x_poses, gridmap, map_res)

    # Count beams that land outside the map
    x_idx = z_in_map[:, 0, :]
    y_idx = z_in_map[:, 1, :]
    rows, cols = likelihood_map.shape
    inside = (x_idx >= 0) & (x_idx < cols) & (y_idx >= 0) & (y_idx < rows)

    # Compute probabilty for those outside beam occurences.
    beam_probs = np.full((B, ranges.size), p_outside, dtype=float)

    # Obtain probabilties from beams that end within the gridmap

    p_ij = likelihood_map[y_idx[inside], x_idx[inside]]

    # Compute final weight by combining both subsets of beams.
    beam_probs[inside] = np.maximum(p_ij, 1e-12)
    weights = np.prod(beam_probs, axis=1)


    ##END_STUDENT_CODE
    shape_check("weights",weights,(B,))
    return weights

def low_variance_resampler_parallel(particles, weights):
    ##STUDENT_CODE: #TODO Q3: Implement the low variance resampling strategy
    N = particles.shape[0]
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    # Compute random resample_offset in range [0:1/N)
    resample_offset = float(np.random.uniform(0, 1 / N))

    # N evenenly distributed bins [0,1) shifted by offset_start
    positions = resample_offset + np.arange(N) / N

    # Find first index of cummulative weights that overlay the left bin border
    resampled_idx = np.searchsorted(np.cumsum(weights), positions, side="left")
    resampled_idx = np.clip(resampled_idx, 0, N - 1)

    # DO not loop over number of bins!
    resampled_particles = particles[resampled_idx]

    ##END_STUDENT_CODE
    shape_check("resampled_particles",resampled_particles,(N,3))
    shape_check("resampled_idx",resampled_idx,(N,))
    shape_check("resample_offset",resample_offset,float)
    return resampled_particles, resampled_idx, resample_offset

#######################################################
#   Transformation helper functions - Do not modify!  #
#######################################################


def world2map(pose, gridmap, map_res):
    """Trasnform world coordinates to discrete map coordinates"""
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    """Transform pose vector pose v = [x, y, θ] into a 2D homogeneous transformation matrix t(3x3)."""
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    """Transform 2D homogeneous transformation matrix t(3x3) into pose vector pose v = [x, y, θ] into a ."""
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    """Converts ranges and angles to homogeneous points."""
    # 2D points
    points = np.array(
        [np.multiply(ranges, np.cos(angles)), np.multiply(ranges, np.sin(angles))]
    )
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    """Computes the endpoints of raw scans into map coordinates.
    Returns: Numpy array with x,y coordinates for each observation"""

    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose

def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


##############################################
#   Random helper functions - Do not modify!  #
##############################################


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 3))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map, 1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map, 0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi

    return particles


def get_sample(std):
    # Irwin–Hall -> Using Central Limit Theorem to generate cheap normal distributed samples.
    tot = 0
    for i in range(12):
        tot += np.random.uniform(-std, std)

    return 0.5 * tot


##################################################
#   Debugging helper functions - Do not modify!  #
##################################################

def shape_check(tag,obj,target_shape):

    if target_shape==(): 
        if isinstance(obj,np.generic):
            if obj.shape!=target_shape:
                assert False,(EMSG(call_function()+"\n"+f"{tag} should be a scalar!"))
        elif not isinstance(obj, (int, float, complex, bool)):
            assert False,(EMSG(call_function()+"\n"+f"{tag} should be a scalar!"))

    elif isinstance(obj,np.ndarray):
        if obj.shape!=target_shape:
            assert False,(EMSG(call_function()+"\n"+f"{tag}.shape is {obj.shape} but should be ({target_shape},)!"))

RED = "\033[1;31m"
RESET = "\033[0m"

def EMSG(msg)->str:
    return RED + msg + RESET

def call_function(steps: int = 2):
    frame = inspect.currentframe()
    for _ in range(steps):
        if frame is None:
            break  # reached the top frame

        frame = frame.f_back

    if frame is None:
        return f"No frame found"
    else:
        return f'Wrong outputshape in {frame.f_code.co_name}:'


##############################################
#   Plot helper functions - Do not modify!  #
##############################################


def plot_observation(z):
    points = np.array(
        [np.multiply(z[0, :], np.cos(z[1, :])), np.multiply(z[0, :], np.sin(z[1, :]))]
    )
    plt.scatter(points[0, :], points[1, :], marker="x")
    plt.scatter(0, 0, marker="o", color="red")
    plt.show()


def plot_particles(particles, img_map, map_res, s, legend=None, size=None):
    ax = plt.gca()
    ax.matshow(255 - img_map, cmap="Greys", origin="upper")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    ax.plot(xs, ys, s, label=legend, markersize=size)
    if legend is not None:
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=1)
    ax.set_xlim(0, np.size(img_map, 1))
    ax.set_ylim(0, np.size(img_map, 0))
    ax.xaxis.tick_bottom()


def plot_low_variance_resampler(weights, num_particles, resampled_idx, offset):
    base_colors = matplotlib.colormaps["tab20"]
    colors = [base_colors(i % 10) for i in range(num_particles)]

    w_sum = np.cumsum(weights)

    w_offset = offset
    w_bin = np.linspace(0 + w_offset, 1 + w_offset, num_particles + 1)

    ax = plt.gca()

    rect = Rectangle(
        xy=(0, -1.9),
        width=1 / num_particles + 1,
        height=0.8,
        facecolor="black",
        edgecolor="black",
    )
    ax.add_patch(rect)

    plt.text(
        0 + 0.2 / num_particles, -1.7, "Random\n Offset", color="red", size=8, zorder=1
    )
    plt.text(
        1 + 0.2 / num_particles, -1.7, "Random\n Offset", color="red", size=8, zorder=1
    )

    for m in range(num_particles):
        rect = Rectangle(
            xy=(w_sum[m], m - 0.5),
            width=-weights[m],
            height=1,
            facecolor=colors[m],
            edgecolor="black",
            zorder=10,
        )
        ax.add_patch(rect)

        indices = np.where(resampled_idx == m)[0]

        for i in indices:
            plt.plot([w_bin[i]], [m], "x", color="black")
            plt.plot([w_bin[i]], [-1.5], "x", color="black")

            plt.plot(
                [w_bin[i], w_bin[i]], [-1, m - 0.5], color=colors[m], linestyle="--"
            )
            rect = Rectangle(
                xy=(w_bin[i], -2),
                width=w_bin[i + 1] - w_bin[i],
                height=1,
                facecolor=colors[m],
                edgecolor="black",
            )
            ax.add_patch(rect)

    plt.xlim(-0.05, 1.05 + 1 / num_particles)

    special_positions = [-1.5] + list(range(0, num_particles))
    special_labels = ["Resampled \nParticles  "] + [
        f"ID: {i}" for i in range(num_particles)
    ]

    plt.yticks(special_positions, special_labels)
    plt.show()


def plot_mc_localization(
    GifWriter, init_particles, particles, img_map, map_res, title, legend, s
):
    fig = plt.gcf()
    plt.clf()
    plt.title(title)
    size = get_marker_size(particles.shape[0])
    fig.subplots_adjust(left=0.15, right=0.90, bottom=0.0, top=1.2)
    plot_particles(
        init_particles, img_map, map_res, ".r", legend="Inital particles", size=size
    )
    plot_particles(particles, img_map, map_res, s, legend=legend, size=size)
    plt.grid(color="gray", linestyle="--", linewidth=0.7)
    fig.canvas.draw()
    GifWriter.grab_frame()


def get_marker_size(n_points):
    if n_points <= POINTSIZE_TRESHHOLDS[0][0]:
        return POINTSIZE_TRESHHOLDS[0][1]
    if n_points >= POINTSIZE_TRESHHOLDS[-1][0]:
        return POINTSIZE_TRESHHOLDS[-1][1]

    for i in range(len(POINTSIZE_TRESHHOLDS) - 1):
        x0, y0 = POINTSIZE_TRESHHOLDS[i]
        x1, y1 = POINTSIZE_TRESHHOLDS[i + 1]
        if x0 <= n_points <= x1:
            size = y0 + (n_points - x0) * (y1 - y0) / (x1 - x0)
            return size


def init_plot_mc_localization(num_particles, VECTORIZED):
    fig, ax = plt.subplots(figsize=(3, 5))
    GifWriter = PillowWriter_CUSTOM(fps=5, loop=None, fig=fig)

    if VECTORIZED is False:
        msg = "MCL"
    else:
        msg = "VectorizedMCL"
    file_name = f"{msg}_p={num_particles}.gif"
    return file_name, fig, GifWriter


class PillowWriter_CUSTOM(PillowWriter):
    def __init__(self, fps=10, loop=0, fig=None):
        super().__init__(fps=fps)
        self.loop = loop
        self._frames = []
        self._fig = fig

    def finish(self):
        if len(self._frames) != 0:
            self._frames[0].save(
                self.outfile,
                save_all=True,
                append_images=self._frames[1:],
                duration=int(1000 / self.fps),
                loop=self.loop,
            )


def plot_parallelization_comparison(
    start_particles, total_steps, scale_factor=4,reps_each=1, tsteps=10, return_data=False, surpress_pb=False
):
    noise = [0.1, 0.1, 0.1, 0.1]
    odom = [0.0, 0.1, 1.0]

    data = pickle.load(
        open(
            "dataset_mit_csail.p",
            "rb",
        )
    )
    gridmap = 255 - data["img_map"]
    likelihood_map = data["likelihood_map"]
    odom = data["odom"][:tsteps]
    z = data["z"][:tsteps]
    img_map = data["img_map"]
    map_res = 0.01

    iterations = [start_particles * scale_factor * i for i in range(1, total_steps + 1)]

    data_plot = {"par": [], "nonpar": [], "num_par": []}

    
    for num_particles in iterations:
        if not surpress_pb:
            progress_bar(num_particles, iterations[-1])
        for rep in range(reps_each):     
            particles = init_uniform(num_particles, data["img_map"], map_res)
            t1 = timeit.default_timer()
            mc_localization(
                odom,
                z,
                num_particles,
                particles,
                noise,
                gridmap,
                likelihood_map,
                map_res,
                img_map,
                viz_timestep_intervall=None,
                parallel_mode=False,
                surpress_pb=True
            )
            t2 = timeit.default_timer()
            mc_localization(
                odom,
                z,
                num_particles,
                particles,
                noise,
                gridmap,
                likelihood_map,
                map_res,
                img_map,
                viz_timestep_intervall=None,
                parallel_mode=True,
                surpress_pb=True
            )
            t3 = timeit.default_timer()

            nonpar = t2 - t1
            par = t3 - t2
            data_plot["par"].append(par)
            data_plot["nonpar"].append(nonpar)
            data_plot["num_par"].append(num_particles)
        # print("-"*100)
    if not surpress_pb:
        clear_output(wait=True)
        

    plt.close("all")
    x = data_plot["num_par"]

    x_unique = np.unique(data_plot["num_par"])
    y_means_par = [
        np.mean([y for x, y in zip(data_plot["num_par"], data_plot["par"]) if x == xi])
        for xi in x_unique
    ]
    y_means_nonpar = [
        np.mean(
            [y for x, y in zip(data_plot["num_par"], data_plot["nonpar"]) if x == xi]
        )
        for xi in x_unique
    ]

    # Fit linear trend lines
    par_fit = np.polyfit(x_unique, y_means_par, 1)  # degree 1 = linear
    nonpar_fit = np.polyfit(x_unique, y_means_nonpar, 1)

    # Generate y-values from the fit
    par_trend = np.polyval(par_fit, x)
    nonpar_trend = np.polyval(nonpar_fit, x)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.title("Scaling Comparison")
    plt.scatter(x, data_plot["par"], marker="x", c="orange", label="Parallelized")
    plt.plot(x, par_trend, "--", c="orange")
    plt.scatter(x, data_plot["nonpar"], marker="x", c="blue", label="Non-parallelized")
    plt.plot(x, nonpar_trend, "--", c="blue")
    plt.xlabel("Index")
    plt.ylabel("RunTime (s)")
    plt.legend(loc="upper left")

    ax = plt.gca()
    ax2 = ax.twinx() 
    y_improve = [nonpar/par for par, nonpar in zip(par_trend,nonpar_trend)]
    ax2.plot(x,y_improve, "--", c="green",label="nonpar/par")
    plt.ylabel("Runtime Decrease Factor")
    
    plt.legend(loc="upper right")
    plt.grid(True)

    if return_data:
        return y_improve, data_plot
    else:
        plt.savefig("ScalingOfMCL")
        plt.show()

def progress_bar(iteration, total, bar_length=30):
    percent = iteration / total
    filled_len = int(bar_length * percent)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)
    clear_output(wait=True)  # update in-place
    
    print(f'Progress: |{bar}| {percent*100:.1f}%')
