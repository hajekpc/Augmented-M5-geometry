# -*- coding: utf-8 -*-
"""
#==============================================================================
#== @TITLE:      Augmented M5 GEOMETRY
#== @AUTHOR:     Petr Hajek
#== @DATE:       2021-11-16
#== @LICENSE:    MPL-2.0 license
#== @NOTES:      Creates Augmented M5 geometry based on real STL data
#== @REFERENCES: [1] SCHERER, Ronald C. et al. (2001) Intraglottal pressure 
                 profiles for a symmetric and oblique glottis 
                 with a divergence angle of 10 degrees.
#==============================================================================
"""

#==============================================================================
#== CLEAR VARIABLES
#==============================================================================
from IPython import get_ipython
get_ipython().magic('reset -sf')


#==============================================================================
#== Input packages
#==============================================================================
import numpy as np
# import math as m
from matplotlib import pyplot as plt
import random

from scipy import optimize
# from scipy.optimize import least_squares

# import trimesh
from stl import mesh

from mpl_toolkits import mplot3d


#==============================================================================
#==============================================================================
#== Of what the script consists and how it works
#==============================================================================
#==============================================================================

#== (STL_0) Functions for clockwise and counterclockwise rotations in 3D, etc
#== (STL_1) Func. read_STL: reads STL object and saves coordinates of points
#== (STL_2) Func. rotffset_STL: rotates and offsets the points
#== (STL_3) Func. slice_STL: saves STL slices and its positions

#== (M5_0) Functions for clockwise and counterclockwise rotation in 2D
#== (M5_1) Func. M5: model is created as a function of the Scherer's parameters
#== (M5_2) Func. M5_fit: model is transformed to the fitting function 
#== (M5_3) Func. M5_rot: to find correct rotation of measured data that matches the fitting function
#== (M5_4) Func. M5_fitting: fits the measured data with M5_fit and returns basic statistics

#== So: 
#   The functions work together as a pipe. The output from one is the input 
#   for the next, so one can tune proper parameters of particular function
#   and then can pass output to the next function. That's convenient!

#== The pipe with STL functions is as follows: 
#   Copy your STL to your working folder: stl-filename.stl -->
#   --> VFmesh, VFmesh_clean = read_STL('stl-filename.stl') -->
#   --> VFmesh_rotffset = rotffset_STL(VFmesh_clean) -->
#   --> VFslices, VFslices_x = slice_STL(VFmesh_rotffset, n_slices = 10, xtol = 0.1)

#== The pipe with M5 functions fitting is as follows:
#   Now you have VFslices which have to be rotated, then can be fitted -->
#   --> VFslices_rot, p0 = M5_rot(VFslices, optional arguments go here) -->
#   --> popt, pcov, perr, R_sq, popt_legend = M5_fitting(VFslices_rot, p0, n_slice)
#   From the last function, you obtain: 
#       Optimal M5 model parameters (popt), 
#       Parameters of covariance (pcov),
#       Standard deviation errors on the parameters (perr),
#       R square reliability of the fit (R_sq),
#       Description of the optimal parametrs (popt_legend).
#   Based on this, you can create (and plot) the fitted M5 model, but
#   script will do it automatically :D.


#== Notes:
#   Details are given inside each function


"""
#==============================================================================
#==============================================================================
#==============================================================================
#== STL PART
#==============================================================================
#==============================================================================
#==============================================================================
"""

#==============================================================================
#== (STL_0) 3D Rotations
#==============================================================================

def rot3D_cw(a, alpha = 90, beta = 90, gamma = 90):
    #== Function input:
    # a ................... [m] numpy.array of point coordinates to rotate
    # alpha, beta, gamma .. [deg] clockwise angles from unrotated x, y, z
    
    alpha = np.radians(alpha)
    beta  = np.radians(beta)
    gamma = np.radians(gamma)
    
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    
    rotx = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]])
    roty = np.array([[cb, 0, -sb], [0, 1, 0], [sb, 0, cb]])
    rotz = np.array([[cg, sg, 0], [-sg, cg, 0], [0, 0, 1]])
    
    rotxyz = rotz @ roty @ rotx
    a_rot = np.matmul(rotxyz, a)
    return a_rot


def rot3D_ccw(a, alpha = 90, beta = 90, gamma = 90):
    #== Function input:
    # a ................... [m] numpy.array of point coordinates to rotate
    # alpha, beta, gamma .. [deg] counterclockwise angles from unrotated x, y, z
    
    alpha = np.radians(alpha)
    beta  = np.radians(beta)
    gamma = np.radians(gamma)
    
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    
    
    rotx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    roty = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rotz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    
    rotxyz = rotz @ roty @ rotx
    a_rot = np.matmul(rotxyz, a)
    return a_rot


#==============================================================================
#== (STL_0) How to make axes equal in 3D plot of STL
#==============================================================================

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    # Set 3D plot axes to equal scale.
    # Make axes of 3D plot have equal scale so that spheres appear as
    # spheres and cubes as cubes. Required since `ax.axis('equal')`
    # and `ax.set_aspect('equal')` don't work on 3D.
    
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])




#==============================================================================
#==============================================================================
#== (STL_1) Read STL [using numpy-stl] and extract STL points to numpy array
#==============================================================================
#==============================================================================

def read_STL(STL_filename = 'stl-filename.stl'):
    #== INPUT
    # STL_filename which has to reside in your workplace
    # (This uses numpy-stl package)
    # (Also PyMesh exists and trimesh or stl-slicer)
    
    #== OUTPUT
    # VFmesh ......... numpy-stl object
    # VFmesh_clean ... coordinates of STL points (only unique points)
    
    
    #== FUNCTION
    # Read STL
    VFmesh = mesh.Mesh.from_file(STL_filename)
    
    # Save all points of the raw STL
    VFmesh_raw = VFmesh.vectors.reshape([int(VFmesh.vectors.size/3), 3])
    
    # Save only unique points
    VFmesh_clean = np.unique(VFmesh_raw, axis=0)
    
    
    #== PLOT
    fig = plt.figure()
    ax = mplot3d.Axes3D(fig)
    
    # Add the vectors to the plot
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(VFmesh.vectors))
    
    # Auto scale to the mesh size
    scale = VFmesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    ax.set_xlim3d(min(VFmesh_clean[0]), max(VFmesh_clean[0])) # Same as ax.auto_scale_xyz(scale, scale, scale)
    ax.set_ylim3d(min(VFmesh_clean[1]), max(VFmesh_clean[1])) # Same as ax.auto_scale_xyz(scale, scale, scale)
    ax.set_zlim3d(min(VFmesh_clean[2]), max(VFmesh_clean[2])) # Same as ax.auto_scale_xyz(scale, scale, scale)
    
    # Manual scale to the mesh size
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(0, 20)
    ax.set_zlim3d(360, 380)
    
    # Plot settings
    ax.set_title('STL_1 | Raw STL mesh')
    ax.set_xlabel(r'$x$ [$\mathrm{mm}$]')
    ax.set_ylabel(r'$y$ [$\mathrm{mm}$]')
    ax.set_zlabel(r'$z$ [$\mathrm{mm}$]')
    
    # Show the plot to the screen and save PDF
    plt.savefig('3D_STL_01-raw-mesh.pdf', dpi=600, transparent=True, bbox_inches = 'tight')
    plt.show()

    return VFmesh, VFmesh_clean




#==============================================================================
#==============================================================================
#== (STL_2) Rotate and offset the VF points to the third quadrant in xy plane
#==============================================================================
#==============================================================================

def rotffset_STL(VFmesh_clean, Rx = 80, Ry = 19, Rz = 11):
    #== INPUT
    # VFmesh_clean ... coordinates of STL points from read_STL function
    # Rx, Ry, Rz ..... [deg] clockwise rotation angles
    
    #== OUTPUT
    # VFmesh_rotffset ... rotated and offset STL points
    # Note: offset is count automatically from the maximum of coordinates
    
    
    #== FUNCTION
    # Transpose VF points
    VFmesh_trans = np.transpose(VFmesh_clean)
    
    # Rotate transposed points
    VFmesh_rot = []
    
    for i in range(len(VFmesh_trans[0,:])):
        VFmesh_rot.append(rot3D_cw(VFmesh_trans[:,i], Rx, Ry, Rz))
    
    VFmesh_rot = np.array(VFmesh_rot, dtype = 'float')
    
    # Offset rotated points
    x_max = np.max(VFmesh_rot[:,0])
    y_max = np.max(VFmesh_rot[:,1])
    z_max = np.max(VFmesh_rot[:,2])
    
    x_offset = np.transpose(np.array([VFmesh_rot[:,0] - x_max]))
    y_offset = np.transpose(np.array([VFmesh_rot[:,1] - y_max]))
    z_offset = np.transpose(np.array([VFmesh_rot[:,2] - z_max]))
    
    # Pack the arrays together
    VFmesh_rotffset = np.hstack((x_offset, y_offset, z_offset))
    
    
    #== PLOT
    fig = plt.figure(figsize=(10,10))
    
    # Set up the axes for the first plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Scatter plot
    ax.scatter(VFmesh_rotffset[:,0], VFmesh_rotffset[:,1], VFmesh_rotffset[:,2], 
                marker = 'o', edgecolor='r', linewidth=0, facecolor='r', s = 0.5)
    ax.scatter(0, 0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
    
    # Plot settings
    ax.set_title('STL_2 | Rotated STL points', fontsize = 15)
    ax.set_xlabel(r'$x$ [$\mathrm{mm}$]')
    ax.set_ylabel(r'$y$ [$\mathrm{mm}$]')
    ax.set_zlabel(r'$z$ [$\mathrm{mm}$]')
    
    # Makes equal axes (plus changes perspective)
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    
    
    # Set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    
    # Scatter plot
    ax.scatter(VFmesh_rotffset[:,0], VFmesh_rotffset[:,1], VFmesh_rotffset[:,2], 
                marker = 'o', edgecolor='r', linewidth=0, facecolor='r', s = 0.5)
    ax.scatter(0, 0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
    
    # Plot settings
    ax.view_init(0,0)
    ax.set_title('YZ plane')
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    
    # Makes equal axes (plus changes perspective)
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    
    
    # Set up the axes for the third plot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    
    # Scatter plot
    ax.scatter(VFmesh_rotffset[:,0], VFmesh_rotffset[:,1], VFmesh_rotffset[:,2], 
                marker = 'o', edgecolor='r', linewidth=0, facecolor='r', s = 0.5)
    ax.scatter(0, 0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
    
    # Plot settings
    ax.view_init(90,90)
    ax.set_title('XY plane')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.zaxis.set_ticklabels([])
    
    # Makes equal axes (plus changes perspective)
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    
    
    # Set up the axes for the fourth plot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Scatter plot
    ax.scatter(VFmesh_rotffset[:,0], VFmesh_rotffset[:,1], VFmesh_rotffset[:,2], 
                marker = 'o', edgecolor='r', linewidth=0, facecolor='r', s = 0.5)
    ax.scatter(0, 0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
    
    # Plot settings
    ax.view_init(0,90)
    ax.set_title('XZ plane')
    ax.set_xlabel(r'$x$')
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel(r'$z$')
    
    # Makes equal axes (plus changes perspective)
    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required
    
    
    # Plot offset
    plt.subplots_adjust(left = 0.1, right=0.9, 
                        bottom=0.1, top = 0.9, 
                        wspace=0.2, hspace=0.3
                        )
    
    
    # Show the plot to the screen and save PDF
    plt.savefig('3D_STL_02-points.pdf', dpi=600, transparent=True, bbox_inches = 'tight')
    plt.show()
    
    return VFmesh_rotffset




#==============================================================================
#==============================================================================
#== (STL_3) Save STL slices and their positions
#==============================================================================
#==============================================================================

def slice_STL(VFmesh_rotffset, n_slices = 10, xtol = 0.1):
    #== INPUT
    # VFmesh_rotffset ... points from rotffset_STL function
    # n_slices .......... number of slices to create
    # x_tol ............. picking tolerance around the slice position
    # (Good starting point for x_tol is scanner resolution: 0.1 mm, default)
    # (Model has to be sliced by YZ plane, so crucial is proper model rotation)
    
    #== OUTPUT
    # VFslices ... coordinates of picked points from all slices
    # VFslices_x ... x coordinate of YZ plane which is used to slice the model
    
    
    #== FUNCTION
    # Find slices
    x_max = np.max(VFmesh_rotffset[:,0])
    x_min = np.min(VFmesh_rotffset[:,0])
    
    VFslices_x = np.linspace(x_min, x_max, num = n_slices, endpoint = False)
    
    # Find and save coordinates of slices to list of numpy arrays
    VFslices = []
    for i in range(len(VFslices_x)):
        # Find indices of picked coordinates
        VFslice_indices = np.where(np.logical_and
            (
            VFmesh_rotffset[:,0]>= VFslices_x[i] - xtol, 
            VFmesh_rotffset[:,0]<= VFslices_x[i] + xtol)
            )
        
        # Find coordinates of the slice
        VFslice = []
        for j in range(len(VFslice_indices)):
            VFslice.append(VFmesh_rotffset[VFslice_indices[j],:])
            VFslice = VFslice[0]
        
        # Save picked coordinates of all slices
        VFslices.append(VFslice.copy())
    
    
    #== PLOT
    fig = plt.figure(figsize=(10,25))
    plt.title("STL_3 | Slices \n\n", fontsize = 15)
    plt.axis('off')
    
    for i in range(len(VFslices_x)):
        # Set up the subplot layout
        ax = fig.add_subplot(int(len(VFslices_x)/2), int(len(VFslices_x)/5), i+1)
        
        # Plot measured data
        plt.plot(VFslices[i][:,1], VFslices[i][:,2], '.r', markersize = 2.5, markeredgewidth = 1)
        ax.scatter(0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
        
        # Plot properties
        plt.title(r"Slice " + str(i) + " in $x$ = " + format(VFslices_x[i], ".3f") + " mm")
        plt.xlabel(r'$y$ [$\mathrm{mm}$]')
        plt.ylabel(r'$z$ [$\mathrm{mm}$]')
        plt.minorticks_on()
        plt.grid(b=True, which='major', linestyle='-', alpha = 0.50)
        plt.grid(b=True, which='minor', linestyle=':', alpha = 0.25)
        plt.axis('equal')
        plt.xlim(np.min(VFmesh_rotffset[:,1]), 0.2)
        plt.ylim(np.min(VFmesh_rotffset[:,2]), 0.2)
    
    # Plot offset
    plt.subplots_adjust(left = 0.1, right=0.7,
                        bottom=0.2, top = 0.8,
                        wspace=0.4, hspace=0.4
                        )
    
    plt.savefig('3D_STL_03-slices.pdf', dpi=600, transparent=True, bbox_inches = 'tight')
    plt.show()
    
    return VFslices, VFslices_x









"""
#==============================================================================
#==============================================================================
#==============================================================================
#== M5 PART
#==============================================================================
#==============================================================================
#==============================================================================
"""

#==============================================================================
#== (M5_0) Rotations
#==============================================================================

def rot_cw(a, alpha = 110):
    #== Function input:
    # a ....... numpy.array of point coordinates to rotate
    # alpha ... [deg] clockwise angle
    
    alpha = np.radians(alpha)
    c, s = np.cos(alpha), np.sin(alpha)
    rot = np.array([[c, s], [-s, c]])
    
    a_rot = np.matmul(rot, a)
    return a_rot

def rot_ccw(a, alpha = 110):
    #== Function input:
    # a ....... numpy.array of point coordinates to rotate
    # alpha ... [deg] counterclockwise angle
    
    alpha = np.radians(alpha)
    c, s = np.cos(alpha), np.sin(alpha)
    rot = np.array([[c, -s], [s, c]])
    
    a_rot = np.matmul(rot, a)
    return a_rot




#==============================================================================
#==============================================================================
#== (M5_1) M5 model as a function
#==============================================================================
#==============================================================================

def M5(R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG = 40, division = int(100), rotation = 0):
    #== Creates M5 geometry as a function of R_O, T, PSI according to:
    # [1] SCHERER, Ronald C. et al. (2001) Intraglottal pressure profiles 
    # for a symmetric and oblique glottis with a divergence angle of 10 degrees.
    
    #== Function input:
    # R_O ...                   # [m] 1st official M5 variable 
    # T .....                   # [m] 2nd official M5 variable 
    # PSI ...                   # [rad] 3rd official M5 variable
    # D_VF ..                   # [m] VF depth in X direction
    # W_G ...                   # [m] Glottal width (Offset of the whole model in x axis)
    # D_Y ...                   # [m] Offset of the whole model in y axis
    # R_A ...                   # [m] Upper radius instead of the horizontal line between A and B
    # alpha ...                 # [deg] The angle of upper arc of the radius R_A
    # R_E ...                   # [m] Lower radius instead of the oblique line between E and F
    # beta ...                  # [deg] The angle of lower arc of the radius R_E
    # VT_ANG = 40               # [deg] 1st unofficial M5 constant; Default Scherer's value: 40 deg
    # division = int(100)       # Division of the one line (eg. from A to B)
    # rotation = 0              # [deg] Clockwise rotation, depends on function rot_cw()
    
    #== Function output:
    # Points = (('A',AX,AY), ('B',BX,BY), ('C',CX,CY), ('D',DX,DY), ('E',EX,EY), ('F',FX,FY))
    # Centers = (('G',GX,GY), ('H',HX,HY))
    # Lines = ( (Line1), ...) = ( ((x coordinates), (y coordinates)), ...)
    # Dimensions = (('R_PSI', R_PSI), ('R_L', R_L), ('R40', R40), ('B', B), ('Q1', Q1), ('Q2', Q2), ('Q3', Q3), ('Q4', Q4), ('Q5', Q5))
    
    #== Dependencies (what functions are inside this one):
    # rot_cw
    
    
    #== Prepare variabls
    PSI = np.radians(PSI)
    VT_ANG = np.radians(VT_ANG)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    
    #== M5 equations (according to [1])
    R_PSI = R_O / (1 - np.sin(PSI/2))
    R_L = T/2
    R40 = T/2
    B = (2**(1/2))*R_PSI / ((1 + np.sin(PSI/2))**(1/2))
    
    Q1 = (T - R_O - (T/2)*np.sin(PSI/2)) / (np.cos(PSI/2))
    Q2 = (T/2)*np.sin(PSI/2)
    Q3 = Q1*np.cos(PSI/2)
    Q4 = R_O
    Q5 = (T/2)*np.sin(50)
    
    Dimensions = (('R_PSI', R_PSI), ('R_L', R_L), ('R40', R40), ('B', B), ('Q1', Q1), ('Q2', Q2), ('Q3', Q3), ('Q4', Q4), ('Q5', Q5))
    
    
    #== M5 points (transformed from Scherer's variables and equations)
    AX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2)))) - R_A*np.sin(alpha)
    AY = D_Y + R_A*(1 - np.cos(alpha))
    
    BX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))
    BY = 0 + D_Y
    
    CX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2)))*(1-np.cos(PSI/2)))
    CY = -((R_O / (1 - np.sin(PSI/2)))*(1 + np.sin(PSI/2))) + D_Y
    
    DX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2))
    DY = -(T+(T/2)*np.sin(PSI/2)) + D_Y
    
    EX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2) + (T/2)*np.cos(PSI/2) - (T/2)*np.sin(VT_ANG))
    EY = -(T + (T/2) - (T/2)*(1-np.cos(VT_ANG))) + D_Y
    
    FX = EX + R_E*np.sin(VT_ANG) - R_E*np.cos(np.radians(90)-VT_ANG-beta)
    FY = EY - R_E*np.cos(VT_ANG) + R_E*np.sin(np.radians(90)-VT_ANG-beta)
    
    GX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))
    GY = -((R_O / (1 - np.sin(PSI/2)))) + D_Y
    
    HX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2) + (T/2)*np.cos(PSI/2))
    HY = -(T) + D_Y
    
    IX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))   # BX
    IY = R_A + D_Y
    
    JX = EX + R_E*np.sin(VT_ANG)
    JY = EY - R_E*np.cos(VT_ANG)
    
    #== Rotated M5 points
    A = rot_cw(np.array([AX, AY]), rotation)
    AX, AY = A[0], A[1]
    
    B = rot_cw(np.array([BX, BY]), rotation)
    BX, BY = B[0], B[1]
    
    C = rot_cw(np.array([CX, CY]), rotation)
    CX, CY = C[0], C[1]
    
    D = rot_cw(np.array([DX, DY]), rotation)
    DX, DY = D[0], D[1]
    
    E = rot_cw(np.array([EX, EY]), rotation)
    EX, EY = E[0], E[1]
    
    F = rot_cw(np.array([FX, FY]), rotation)
    FX, FY = F[0], F[1]
    
    G = rot_cw(np.array([GX, GY]), rotation)
    GX, GY = G[0], G[1]
    
    H = rot_cw(np.array([HX, HY]), rotation)
    HX, HY = H[0], H[1]
    
    I = rot_cw(np.array([IX, IY]), rotation)
    IX, IY = I[0], I[1]
    
    J = rot_cw(np.array([JX, JY]), rotation)
    JX, JY = J[0], J[1]
    
    
    #== Save All M5 Points (('Name', X, Y, Z), ...)
    Points = (('A',AX,AY), ('B',BX,BY), ('C',CX,CY), ('D',DX,DY), ('E',EX,EY), ('F',FX,FY))
    Centers = (('G',GX,GY), ('H',HX,HY))
    
    
    #== Save All M5 Lines ( ((x coordinates), (y coordinates)), ...)
    Lines = []
    for i in range(0,len(Points)-1):
        if (
                (Points[i][0] == 'A' and Points[i+1][0] == 'B')
             or (Points[i][0] == 'B' and Points[i+1][0] == 'C')
             or (Points[i][0] == 'D' and Points[i+1][0] == 'E')
             or (Points[i][0] == 'E' and Points[i+1][0] == 'F')
             ):
            if Points[i][0] == 'A' and Points[i+1][0] == 'B':
                Radius = R_A
                AngleStart = np.radians(180) + np.radians(rotation)
                AngleEnd = np.radians(180) + alpha + np.radians(rotation)
                x_offset = IX
                y_offset = IY
            if Points[i][0] == 'B' and Points[i+1][0] == 'C':
                Radius = np.sqrt((Points[i][1] - GX)**2 + (Points[i][2] - GY)**2)   # R_{PSI}
                AngleStart = 0 + np.radians(rotation)
                AngleEnd = np.radians(90 + np.degrees(PSI)/2) + np.radians(rotation)
                x_offset = GX
                y_offset = GY
            if Points[i][0] == 'D' and Points[i+1][0] == 'E':
                Radius = np.sqrt((Points[i][1] - HX)**2 + (Points[i][2] - HY)**2)   # R_{L}
                AngleStart = np.radians(90 + np.degrees(PSI)/2) + np.radians(rotation)
                AngleEnd = np.radians((90 + np.degrees(PSI)/2) + (90 - np.degrees(PSI)/2 - np.degrees(VT_ANG))) + np.radians(rotation)
                x_offset = HX
                y_offset = HY
            if Points[i][0] == 'E' and Points[i+1][0] == 'F':
                Radius = R_E
                AngleStart = -VT_ANG +np.radians(rotation)
                AngleEnd = -VT_ANG -beta +np.radians(rotation)
                x_offset = JX
                y_offset = JY
            Arc_res = np.linspace(AngleStart,AngleEnd,division)
            Lines.append(
                (
                tuple(Radius*np.sin(j)+x_offset for j in Arc_res), 
                tuple(Radius*np.cos(j)+y_offset for j in Arc_res)
                )
                )
        else:
            Lines.append(
                (
                tuple(np.linspace(Points[i][1], Points[i+1][1], num=division)), 
                tuple(np.linspace(Points[i][2], Points[i+1][2], num=division))
                )
                )
    
    
    #== Return output
    return Points, Centers, Lines, Dimensions




#==============================================================================
#==============================================================================
#== (M5_2) M5 as a fitting function (created from the partial fitting functions)
#==============================================================================
#==============================================================================

def M5_fit(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
    #== How it works in 4 easy steps
    # (1) Computes M5 points based on the original Scherer's equations 
    # (which are transformed to be able to compute points coordinates directly)
    # (2) Points coordinates are rotated clockwise 110 deg 
    # (which transforms M5 model into a function nearly symmetrical about y axis)
    # (3) Based on the rotated points, the lines are created as a functions
    # (lines are y = a*x + b; circular sections are from Pythagorean theorem)
    # (4) The lines are supplied to one piecewise function
    # (then the model geometry is y = f(x); and also there are constants)
    
    #== Input: What it needs
    # x ..... # Domain of the fitting function (can be (-np.inf, np.inf))
    # R_O ... # [m] 1st official M5 variable 
    # T ..... # [m] 2nd official M5 variable 
    # PSI ... # [rad] 3rd official M5 variable
    # D_VF .. # [m] VF depth in X direction
    # W_G ... # [m] Glottal width (Twice the offset of the whole model in x axis)
    # D_Y ... # [m] Offset of the whole model in y axis
    # R_A ... # [m] Upper radius instead of the horizontal line between A and B
    # alpha . # [deg] The angle of upper arc of the radius R_A
    # R_E ... # [m] Lower radius instead of the oblique line between E and F
    # beta .. # [deg] The angle of lower arc of the radius R_E
    # VT_ANG  # [deg] 1st unofficial M5 constant; Default Scherer's value: 40 deg
    
    #== Output: What you get
    # You get M5 geometry as a y = f(x) 
    # rotated clockwise 110 degrees to make VF symmetrical about y axis
    
    #== Dependencies (what functions are inside this one):
    # rot_cw
    
    
    #== (0) Prepare data
    PSI = np.radians(PSI)
    VT_ANG = np.radians(VT_ANG)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    
    #== (1) M5 points (transformed from Scherer's variables and equations)
    #alpha = np.arcsin((D_VF -(W_G/2+(R_O/(1-np.sin(PSI/2)))) + W_G/2)/R_A)
    AX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2)))) - R_A*np.sin(alpha)
    AY = D_Y + R_A*(1 - np.cos(alpha))
    
    BX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))
    BY = 0 + D_Y
    
    CX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2)))*(1-np.cos(PSI/2)))
    CY = -((R_O / (1 - np.sin(PSI/2)))*(1 + np.sin(PSI/2))) + D_Y
    
    DX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2))
    DY = -(T+(T/2)*np.sin(PSI/2)) + D_Y
    
    EX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2) + (T/2)*np.cos(PSI/2) - (T/2)*np.sin(VT_ANG))
    EY = -(T + (T/2) - (T/2)*(1-np.cos(VT_ANG))) + D_Y
    
    FX = EX + R_E*np.sin(VT_ANG) - R_E*np.cos(np.radians(90)-VT_ANG-beta)
    FY = EY - R_E*np.cos(VT_ANG) + R_E*np.sin(np.radians(90)-VT_ANG-beta)
    
    GX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))
    GY = -((R_O / (1 - np.sin(PSI/2)))) + D_Y
    
    HX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2) + (T/2)*np.cos(PSI/2))
    HY = -(T) + D_Y
    
    IX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2))))   # BX
    IY = R_A + D_Y
    
    JX = EX + R_E*np.sin(VT_ANG)
    JY = EY - R_E*np.cos(VT_ANG)
    
    #== (2) Rotated M5 points
    rotation = 110
    
    A = rot_cw(np.array([AX, AY]), rotation)
    AX, AY = A[0], A[1]
    
    B = rot_cw(np.array([BX, BY]), rotation)
    BX, BY = B[0], B[1]
    
    C = rot_cw(np.array([CX, CY]), rotation)
    CX, CY = C[0], C[1]
    
    D = rot_cw(np.array([DX, DY]), rotation)
    DX, DY = D[0], D[1]
    
    E = rot_cw(np.array([EX, EY]), rotation)
    EX, EY = E[0], E[1]
    
    F = rot_cw(np.array([FX, FY]), rotation)
    FX, FY = F[0], F[1]
    
    G = rot_cw(np.array([GX, GY]), rotation)
    GX, GY = G[0], G[1]
    
    H = rot_cw(np.array([HX, HY]), rotation)
    HX, HY = H[0], H[1]
    
    I = rot_cw(np.array([IX, IY]), rotation)
    IX, IY = I[0], I[1]
    
    J = rot_cw(np.array([JX, JY]), rotation)
    JX, JY = J[0], J[1]
    
    
    #== (3) Creates line functions supplied to the piecewise function
    #== M5 Line from A to B
    def M5_A2B(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
        Radius = np.sqrt((BX - IX)**2 + (BY - IY)**2)   # R_A
        y = tuple(np.sqrt(Radius**2 - (i-IX)**2) + IY for i in x)
        return y
    
    #== M5 Line from B to C
    def M5_B2C(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
        Radius = np.sqrt((BX - GX)**2 + (BY - GY)**2)   # R_{PSI}
        y = tuple(-np.sqrt(Radius**2 - (i-GX)**2) + GY for i in x)
        return y
    
    #== M5 Line from C to D
    def M5_C2D(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
        a = (CY - DY) / (CX - DX)
        b = CY - a*CX
        y = tuple(a*i + b for i in x)
        return y
    
    #== M5 Line from D to E
    def M5_D2E(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
        Radius = np.sqrt((DX - HX)**2 + (DY - HY)**2)   # R_{L}
        y = tuple(-np.sqrt(Radius**2 - (i-HX)**2) + HY for i in x)
        return y
    
    #== M5 Line from E to F
    def M5_E2F(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG):
        Radius = np.sqrt((EX - JX)**2 + (EY - JY)**2)   # R_E
        y = tuple(np.sqrt(Radius**2 - (i-JX)**2) + JY for i in x)
        return y
    
    
    #== (4) The piecewise function describing the M5 model as y = f(x)
    y = np.piecewise(x, 
                         [
                             (x <= AX) & (x >  BX),
                             (x <= BX) & (x >= CX),
                             (x <  CX) & (x >  DX),
                             (x <= DX) & (x >= EX),
                             (x <  EX) & (x >= FX)
                         ],
                         [
                             lambda x: M5_A2B(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG),
                             lambda x: M5_B2C(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG),
                             lambda x: M5_C2D(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG),
                             lambda x: M5_D2E(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG),
                             lambda x: M5_E2F(x, R_O, T, PSI, W_G, D_Y, R_A, alpha, R_E, beta, VT_ANG),
                             lambda x: np.tan(np.radians(90-rotation)) * x
                         ]
                     )
    return y




#==============================================================================
#==============================================================================
#== (M5_3) Rotate points from STL to match the fitting function
#==============================================================================
#==============================================================================

def M5_rot(VFslices, VFslices_x, Rot = 110, R_O = 0.987, T = 3.000, PSI = -10, W_G = 0.1, D_Y = 0.0, R_A = 10.0, alpha = 90.0, R_E = 20.0, beta = 90.0, VT_ANG = 40.0):
    # User has to tune both the slice points (by rotation) 
    # and the M5 fitting function (by Scherer's parameters and VF dimesions)
    # to achieve that the both ones are simmilar to the most.
    # It produces optimally rotated points which match the tuned fitting function.
    # Note: The M5 fitting function is rotated 110 deg clockwise.
    
    #== INPUT
    # VFslices ... coordinates of picked points from all slices (from slice_STL)
    # VFslices_x . x coordinate of YZ plane which is used to slice the model (from slice_STL)
    # Rotation ... [deg] clockwise rotation of picked points
    # R_O, T, PSI ...... [mm, deg] Scherer's parameters 
    # D_VF, W_G, D_Y ... [mm] VF depth, glottal width, y offset 
    # R_A ... # [m] Upper radius instead of the horizontal line between A and B
    # alpha . # [deg] The angle of upper arc of the radius R_A
    # R_E ... # [m] Lower radius instead of the oblique line between E and F
    # beta .. # [deg] The angle of lower arc of the radius R_E
    # VT_ANG ........... [deg] 1st unofficial Scherer's constant 
    
    #== OUTPUT
    # VFslices_rot ... rotated coordinates of picked points from all slices
    # p0 ............. optimal initial guess for the M5 fit (tuned by user!)
    
    #== Dependencies (what functions are inside this one):
    # rot_cw, M5_fit
    
    
    #== FUNCTION
    # Rotate slices
    VFslices_rot = []
    for i in range(len(VFslices)):
        slice_rot = np.zeros((len(VFslices[i][:,0]), 2))
        for j in range(len(VFslices[i][:,0])):
            slice_rot[j,0], slice_rot[j,1] = rot_cw(np.array([VFslices[i][j,1], VFslices[i][j,2]]), Rot)
        VFslices_rot.append(slice_rot.copy())
    
    # Save parameters tuned by user in input
    PSI = np.radians(PSI)
    VT_ANG = np.radians(VT_ANG)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    p0 = (R_O, T, np.degrees(PSI), W_G, D_Y, R_A, np.degrees(alpha), R_E, np.degrees(beta), np.degrees(VT_ANG))
    
    
    #== PLOT
    fig = plt.figure(figsize=(10,25))
    plt.title("M5_3 | Rotated slices\nClockwise rotation = " + format(Rot, ".0f") + " deg \n", fontsize = 15)
    plt.axis('off')
    
    # Find boundaries
    AX = -(W_G/2 + (R_O / (1 - np.sin(PSI/2)))) - R_A*np.sin(alpha)
    AY = D_Y + R_A*(1 - np.cos(alpha))
    
    EX = -(W_G/2 + (T-(R_O / (1 - np.sin(PSI/2)))+(T/2)*np.sin(PSI/2))*np.tan(PSI/2) + (T/2)*np.cos(PSI/2) - (T/2)*np.sin(VT_ANG))
    EY = -(T + (T/2) - (T/2)*(1-np.cos(VT_ANG))) + D_Y
    FX = EX + R_E*np.sin(VT_ANG) - R_E*np.cos(np.radians(90)-VT_ANG-beta)
    FY = EY - R_E*np.cos(VT_ANG) + R_E*np.sin(np.radians(90)-VT_ANG-beta)
    
    # Rotate boundaries
    rotation = 110
    A = rot_cw(np.array([AX, AY]), rotation)
    AX, AY = A[0], A[1]
    F = rot_cw(np.array([FX, FY]), rotation)
    FX, FY = F[0], F[1]
    
    # Prepare the M5 fitting function
    xM5fit = np.linspace(F[0], A[0], 1000)
    
    yM5fit = []
    for x in xM5fit:
        yM5fit.append(M5_fit(x, R_O, T, np.degrees(PSI), W_G, D_Y, R_A, np.degrees(alpha), R_E, np.degrees(beta), np.degrees(VT_ANG)))
    
    # Start plotting
    for i in range(len(VFslices_x)):
        # Set up the subplot layout
        ax = fig.add_subplot(int(len(VFslices_x)/2), int(len(VFslices_x)/5), i+1)
        
        # Plot fitting function
        plt.plot(xM5fit, yM5fit, 'g--', label='M5 fit')
        
        # Plot measured data
        plt.plot(VFslices_rot[i][:,0], VFslices_rot[i][:,1], '.r', markersize = 2.5, markeredgewidth = 1, label='STL data')
        ax.scatter(0, 0, marker = 'o', edgecolor='b', linewidth=0, facecolor='b', s = 10)
        
        # Plot properties
        plt.title(r"Slice " + str(i) + " in $x$ = " + format(VFslices_x[i], ".3f") + " mm")
        plt.legend(loc='upper right')
        plt.xlabel(r'$y$ [$\mathrm{mm}$]')
        plt.ylabel(r'$z$ [$\mathrm{mm}$]')
        plt.minorticks_on()
        plt.grid(b=True, which='major', linestyle='-', alpha = 0.50)
        plt.grid(b=True, which='minor', linestyle=':', alpha = 0.25)
        plt.axis('equal')
        # plt.xlim(np.min(VFslices_rot[i][:,0]), np.max(VFslices_rot[i][:,0]))
        # plt.ylim(np.min(VFslices_rot[i][:,1]), np.max(VFslices_rot[i][:,1]))
    
    # Plot offset
    plt.subplots_adjust(left = 0.1, right=0.7,
                        bottom=0.2, top = 0.8,
                        wspace=0.4, hspace=0.4
                        )
    
    plt.savefig('3D_M5_03-rotated-slices.pdf', dpi=600, transparent=True, bbox_inches = 'tight')
    plt.show()
    
    return VFslices_rot, p0




#==============================================================================
#==============================================================================
#== (M5_4) Fitting of rotated M5 model
#==============================================================================
#==============================================================================

def M5_fitting(VFslices_rot, p0, n_slice = 8):
    # The function needs one slice from the rotated slices (VFslices_rot), 
    # initial guess of the parameters (p0) and
    # the number of chosen slice to be fitted (n_slice).
    # The fitting itself needs rotated data, 
    # beacuse it is carried out in rotated position,
    # but produces output which is rotated back (plot).
    # Optimal parameters (popt) and statistics are invariant on the rotation.
    
    #== INPUT
    # VFslices_rot ... Rotated slices from M5_rot function
    # p0 ............. Tuned initial guess of parameters from M5_rot
    # n_slice ........ Number of slice (chosen from VFslices_rot) to be fitted
    
    #== OUTPUT
    # popt  ......... Optimal parameters upon which one can create the M5 geom.
    # pcov .......... Covariance matrix of optimal parameters
    # perr .......... Standard deviation errors on the parameters
    # R_sq .......... R square reliability of the fit
    # popt_legend ... Description of popt parameters
    
    #== Dependencies (what functions are inside this one):
    # M5_fit, rot_ccw, M5
    
    
    #== FUNCTION
    # Prepare for fitting
    slice_x = VFslices_rot[n_slice][:,0]
    slice_y = VFslices_rot[n_slice][:,1]
    
    # Fitting is carried out in rotated position
    popt, pcov = optimize.curve_fit(
        M5_fit, 
        xdata = slice_x, 
        ydata = slice_y,
        p0 = p0,
        # bounds = (
        #     (0, -np.inf, -np.inf, 0, -5, -np.inf), 
        #     (np.inf, np.inf, np.inf, 10, 5, np. inf)
        #           )
        )
    
    popt_legend = ['R_O [mm]', 'T [mm]', 'PSI [deg]', 'W_G [mm]', 'D_Y [mm]', 
                   'R_A [mm]', 'alpha [deg]', 'R_E [mm]', 'beta [deg]', 
                   'VT_ANG [deg]']
    
    # Make statistics
    perr = np.sqrt(np.diag(pcov)) # Standard deviation errors on the parameters
    corr_matrix = np.corrcoef(slice_y, M5_fit(slice_x, *popt)) # Correlation matrix
    corr = corr_matrix[0,1] # Slice the matrix for Coefficient of Correlation
    R_sq = corr**2 # Calculate R square
    
    
    #== PLOT
    # Rotated
    slice_x_measured_rotated = slice_x
    slice_y_measured_rotated = slice_y
    
    slice_x_fit_rotated = slice_x
    slice_y_fit_rotated = M5_fit(slice_x, *popt)
    
    # Unrotated
    slice_x_measured = []
    slice_y_measured = []
    for i in range(len(slice_x_measured_rotated)):
        slice_xy_measured = rot_ccw(np.array([slice_x_measured_rotated[i], slice_y_measured_rotated[i]]))
        slice_x_measured.append(slice_xy_measured[0])
        slice_y_measured.append(slice_xy_measured[1])
    
    slice_x_fit = []
    slice_y_fit = []
    for i in range(len(slice_x_fit_rotated)):
        slice_xy_fit = rot_ccw(np.array([slice_x_fit_rotated[i], slice_y_fit_rotated[i]]))
        slice_x_fit.append(slice_xy_fit[0])
        slice_y_fit.append(slice_xy_fit[1])
    # Everything is unrotated now
    
    
    # Save points to plot
    M5_popt_points, M5_popt_centers, M5_popt_Lines, M5_popt_dimensions = M5(popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], VT_ANG = popt[9])
    
    # Plot measured data
    plt.plot(slice_x_measured, slice_y_measured, '.', markersize = 5, markeredgewidth = 1, label='Measured data, slice '+str(n_slice))
    
    # Plot least squares fit
    plt.plot(slice_x_fit, slice_y_fit, 'g.', alpha=0.25,
              label='\nAugmented M5 fit: \n$R_O$ = %5.6f mm\n$T$   = %5.6f mm\n$\Psi$  = %5.6f deg\n\n$W_G$ = %5.6f mm\n$Y_{off}$ = %5.6f mm\n\n$R_{AB}$ = %5.6f mm\n$\\alpha_{AB}$ = %5.6f deg\n$R_{EF}$ = %5.6f mm\n$\\beta_{EF}$ = %5.6f deg\n$\\beta$ = %5.6f deg' % tuple(popt))
    
    # Plot M5 points and lines based on coefficients
    for i in range(len(M5_popt_points)):
        plt.plot(M5_popt_points[i][1],M5_popt_points[i][2],'k*', alpha=0.5, markersize = 6, markeredgewidth = 0.01,
                #label = M5_popt_points[i][0] + ' (' + format(M5_popt_points[i][1], ".6f") + ', ' + format(M5_popt_points[i][2], ".6f") + ')'
                )
    for i in range(len(M5_popt_centers)):
        plt.plot(M5_popt_centers[i][1],M5_popt_centers[i][2],'k*', alpha=0.5, markersize = 6, markeredgewidth = 0.01,
                #label = M5_popt_centers[i][0] + ' (' + format(M5_popt_centers[i][1], ".6f") + ', ' + format(M5_popt_centers[i][2], ".6f") + ')'
                )
    for i in range(len(M5_popt_Lines)):
        plt.plot(M5_popt_Lines[i][0], M5_popt_Lines[i][1], alpha=0.75)
    
    
    # Show plot
    plt.title(r"M5_4 | Fitted 2D M5 model with $R^2$ = " + format(R_sq, ".3f"))
    plt.xlabel(r'$x$ [$\mathrm{mm}$]')
    plt.ylabel(r'$y$ [$\mathrm{mm}$]')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    # plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-9.0, 1.0)
    plt.ylim(-9.0, 1.0)
    plt.xticks(rotation = 0)
    plt.tight_layout()
    plt.savefig('2D_M5_fit_slice'+str(n_slice)+'.pdf', dpi=600, transparent=True, bbox_inches = 'tight')
    
    plt.show()
    
    return popt, pcov, perr, R_sq, popt_legend, M5_popt_points, M5_popt_centers





"""
#==============================================================================
#==============================================================================
#==============================================================================
#== START OF THE PIPE
#==============================================================================
#==============================================================================
#==============================================================================
"""

# The functions are described and commented inside themselves!
# Typical parameters are given inside each function used in the pipe.


#==============================================================================
#== (STL_1) Read STL and plot raw mesh data
#==============================================================================

VFmesh, VFmesh_clean = read_STL('stl-filename.stl')


#==============================================================================
#== (STL_2) Find the best rotation to create the best slices
#==============================================================================
# The glottis should be alligned with X axis 
# and cranial-caudal height of the VF should lie in Z axis
# so lateral-medial width has to be in Y axis.
# The VF has to be placed in negative quadrant,
# glottis should be closer to zero 
# and lateral part should be in more negative side:

# Z
#   |       The VF          
# 0 |
#   | * * * * * *
#   |            *
#   |           *
#   |        *
#-5 |    *
# __|____________________
#   |  -5            0    Y

# Tune it with the Rx, Ry, Rz (clockwise rotations around x, y, z in [deg])

VFmesh_rotffset = rotffset_STL(VFmesh_clean, Rx = 80, Ry = 19, Rz = 11)


#==============================================================================
#== (STL_3) Create as many slices as you need
#==============================================================================
# STL geometry is sliced by the YZ plane, 
# so crucial is proper geometry rotation from the preceding step!

VFslices, VFslices_x = slice_STL(VFmesh_rotffset, n_slices = 10, xtol = 0.1)


#==============================================================================
#== (M5_3) Rotate STL points (for the second time) to match the fitting function
#==============================================================================
# User has to tune both the slice points (by rotation) 
# and the M5 fitting function (by Scherer's parameters and VF dimesions)
# to achieve that the both ones are simmilar to the most.
# It produces optimally rotated points which match the tuned fitting function.
# Note: The M5 fitting function is rotated 110 deg clockwise by default.

 VFslices_rot, p0 = M5_rot(VFslices, VFslices_x, 
                     Rot = 110, 
                     R_O = 1.600, T = 2.000, PSI = -13, 
                     W_G = 2.0, D_Y = -1.0, 
                     R_A = 3.0, alpha = 55.0,
                     VT_ANG = 30.0)
                     R_E = 7.0, beta = 40.0,


#==============================================================================
#== (M5_4) Fit STL points (finally!!!) 
#==============================================================================
# The function needs one slice from the rotated slices (VFslices_rot), 
# initial guess for the parameters (p0) and
# the number of chosen slice to be fitted (n_slice).
# The fitting itself needs rotated data, 
# but produces output which is rotated back (plot).
# Optimal parameters (popt) and statistics are invariant on the rotation.

 popt, pcov, perr, R_sq, popt_legend, points, centers = M5_fitting(VFslices_rot, p0, n_slice = 8)


# And that's it. Totally easy, right?

