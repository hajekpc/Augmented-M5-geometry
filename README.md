# Augmented-M5-geometry

The presented script is a part of the conference paper:

**AUGMENTED M5 GEOMETRY OF HUMAN VOCAL FOLD IN PHONATORY POSITION – PILOT RESULTS**

*Hájek Petr<sup>1</sup>, Horáček Jaromír<sup>2</sup>, Švec Jan G.<sup>3</sup>*

**Abstract:** 

The presented contribution deals with a newly designed parametric planar geometry of the vocal fold (VF)
– the augmented M5 model – which is fitted to a real-shaped human vocal fold. The real shape of the vocal fold
during a phonatory position for 112 Hz was obtained from a plaster cast and was digitized by optical scanning.
The geometry model of the vocal fold surfaces was constructed based on the data from the optical scanner and
the augmented M5 model was fitted to a coronal slice of the selected vocal fold surface. The equations of
the augmented M5 model are explained and its parameters, tuned to the real vocal fold geometry, are provided.
The fitting is done in Python 3.8.5 using the scipy.optimize.curve fit package, which contains non-linear least squares method. 
It is shown that the augmented M5 model fits the real data with coefficient R2
close to 1 and the tuned parameters are in a good agreement with the overall vocal fold dimensions and with
the parameters of the original 2D M5 model.

**Keywords:** 

Vocal fold, M5 geometry, non-linear least squares, curve fitting, phonatory position.

<sup>1</sup> Ing. Petr Hájek, PhD.: Institute of Solid Mechanics, Mechatronics and Biomechanics, Brno University of Technology,
Technická 2896/2, 616 69, Brno, CZ; Voice Research Laboratory, Department of Experimental Physics, Palacký University
Olomouc; 17. listopadu 1192/12, 779 00, Olomouc, CZ; hajek.p@fme.vutbr.cz

<sup>2</sup> Ing. Jaromír Horáček, DSc.: Institute of Thermomechanics of the Czech Academy of Sciences, Dolejškova 1402/5, 182 00,
Prague, CZ; jaromirh@it.cas.cz

<sup>3</sup> Prof. RNDr. Jan G. Švec, PhD. et PhD.: Voice Research Laboratory, Department of Experimental Physics, Palacký University
Olomouc, 17. listopadu 1192/12, 779 00, Olomouc, CZ; jan.svec@upol.cz

---

## The script contains following functions:
- (STL_0) Functions for clockwise and counterclockwise rotations in 3D, etc.
- (STL_1) Func. read_STL: reads STL object and saves coordinates of points
- (STL_2) Func. rotffset_STL: rotates and offsets the points
- (STL_3) Func. slice_STL: saves STL slices and its positions

- (M5_0) Functions for clockwise and counterclockwise rotation in 2D
- (M5_1) Func. M5: model is created as a function of the Scherer's [^1] and new parameters 
- (M5_2) Func. M5_fit: model is transformed to the fitting function 
- (M5_3) Func. M5_rot: to find correct rotation of measured data that matches the fitting function
- (M5_4) Func. M5_fitting: fits the measured data with M5_fit and returns basic statistics

## So: 
The functions work together as a pipe. The output from one is the input 
for the next, so one can tune proper parameters of particular function
and then can pass output to the next function. That's convenient!

### The pipe with STL functions works as follows: 
Copy your STL to your working folder --> stl-filename.stl 

--> `VFmesh, VFmesh_clean = read_STL('stl-filename.stl')` 

--> `VFmesh_rotffset = rotffset_STL(VFmesh_clean)` 

--> `VFslices, VFslices_x = slice_STL(VFmesh_rotffset, n_slices = 10, xtol = 0.1)`

### The pipe with M5 functions works as follows:
Now you have VFslices which have to be rotated, then can be fitted: 

--> `VFslices_rot, p0 = M5_rot(VFslices, optional arguments go here)`

--> `popt, pcov, perr, R_sq, popt_legend = M5_fitting(VFslices_rot, p0, n_slice)`

From the last function, you obtain: 
- Optimal M5 model parameters (`popt`), 
- Parameters of covariance (`pcov`),
- Standard deviation errors on the parameters (`perr`),
- R square reliability of the fit (`R_sq`),
- Description of the optimal parametrs (`popt_legend`).
    
Based on this, you can create (and plot) the fitted Augmented M5 model, but
script will plot it automatically. Details are given inside each function in the python script.


**Literature**
[^1]: SCHERER, Ronald C., Daoud SHINWARI, Kenneth J. DE WITT, Chao ZHANG, Bogdan R. KUCINSCHI a Abdollah A. AFJEH, 2001. Intraglottal pressure profiles for a symmetric and oblique glottis with a divergence angle of 10 degrees. The Journal of the Acoustical Society of America [online]. 109(4), 1616–1630. ISSN 0001-4966. Available from: doi:10.1121/1.1333420

