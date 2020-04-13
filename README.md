

# gPET

> latest update: Mar 31, 2020.   
> Authors: Y. Lai, X. Jia* and Y. Chi*  
> Department of Physics, University of Texas at Arlington, Arlington, TX 76019, USA.  
> Innovative Technology Of Radiotherapy Computation and Hardware (iTORCH) Laboratory, Department of Radiation Oncology,   University of Texas Southwestern Medical Center, Dallas, TX 75390, USA.   
> *Corresponding authors: Yujie.Chi@uta.edu, Xun.Jia@UTSouthwestern.edu. 

Other significant contributors in the development of the code include:
1. Yuncheng Zhong
2. Yiping Shao
3. Mingwu Jin
4. Ananta Chalise

## Statement

This is a GPU-based MC simulation package dedicated for PET simulation, co-developed by reseachers from Dr. Xun Jia's group in the University of Texas Southwestern Medical Center and Dr. Yujie Chi's group in the University of Texas at Arlington. For more details or for citation, please refer to the following publication:
https://iopscience.iop.org/article/10.1088/1361-6560/ab5610/meta   
Caution: In this released version, double precision floating-point numbers are employed for recording time in the entire simulation, including those computations on the GPU-end, to improve accuracy. Correspondingly, the time cost for the simulation is expected to be longer than that reported in the paper listed above.

## Hardware and software requirement

To run the code, a single GPU card supporting CUDA platform is required to be installed.
We have compiled gPET successfully on
- Centos 7 with kernel 3.10.0
- Ubuntu 18.04 with kernel 4.15.0

with CUDA version 10.0. The compilation should also work for CUDA version later than 5.0.

## Usage

### Compilation

```bash
nvcc -dc *.cu
nvcc *.o ./kernal/*.o -o gPET
```

### Execution

```bash
./gPET input_PET.in
```

## Package organization

The gPET package is organized in the following structure:

```
/root folder
  “*.cu” files,  for programming function definition and logic flow control;
  “*.h” files,  for programming function declaration and global memory definition;
  “*.in” file, for the entire simulation configuration
  /input folder, for all user input files, including source, phantom, detector configurations, etc.;
  /data folder, for all predefined, code-running-critical data, e.g., material data, cross section, etc.;
  /output folder, for simulation outputs.
```

The main functions of the code files are as follows. “Constants.h” file defines global constants. “gPETInternal.h” defines all global variables used in the host while “gPET_kernals.h” defines that for the device. The “externCUDA.h” file gives all extern declarations. “gPET.h” file then defines all struct data types and declare functions for both host and device. “main.cu” file contains the main function of the package. “iniDevice.cu”, “initialize.cu” and “detector.cu” contain main initialization functions for device, source and phantom, and detector, respectively. “detector.cu” also defines some other functions related to the detector. In the “gPET.cu” file,  the simulation process is defined. All GPU kernel functions are then wrapped into the “gPET_kernals.o” file. 
A simulation example for a small animal PET detector is given in this released version, with the specific configuration seen in the "input_PET.in"  text file.

## Input file preparation

### (1). Phantom

The user defined phantom information is read into the code through the “void loadPhantom(**)” function in the “initialize.cu” file.
The "mat.dat" is an int32-type binary file, containing the material index in voxelized geometry, following the index list in ./data/*matter file. Notation: the index starts with ‘0’.
The "den.dat" is a float32-type binary file, containing material density in the same voxelized geometry as that used for the "mat.dat" file.

### (2). Source

The user defined source information is read into the code through the “readSource(**)” and “readParticle(**)” function in the “initialize.cu” file.
The source files are of two types, text or binary.
The "source.txt" is a text file, containing the source information for the position emission nuclei, in a voxelized geometry. Source indexes are defined based on that in the “./data/isotopes.txt” file.
The "psf.dat" is a float64-type binary file, specifying the gamma or positron source information in the order of location (x, y, z), global time (t), direction (vx, vy, vz) and kinetic energy (e).

### (3). Detector geometry

The “read_file_ro(**)” function in the “detector.cu” file reads the detector information specified in the text file of "config8.geo".
The extension direction of the panel in the local coordinate requires special attention. “+1” (“-1”) represents the extension of the panel parallel (antiparallel) to the unit vector of the local coordinate system.
More details can be seen in the paper listed above.

### (4). Configuration file

See the “main.cu” file to check how to read the information defined in the “input_PET.in” file. Parser is not designed currently. DO NOT add any extra comment lines in this configuration file.

Check out the commented configuration file below.

```
/* Setting device number for the simulation.*/
device number:
0
/*Setting the noncollinearity for the annihilated photon pair. The deviation from antiparallel is described as a Gaussian distribution, with the mean value of zero and the standard deviation sigma from user input. A sigma value of zero means antiparallel.*/
noncolinear angle (sigma of Gaussian in rad):
0.0037056
/*phantom information*/
phantom dimension:
200 200 200
phantom offset in global coordinate (in cm):
-0.5 -0.5 -0.5
phantom size (in cm):
1 1 1
/*phantom data*/
phantom material data file:
input/cylinder_phantom_mat.dat
phantom density data file:
input/cylinder_phantom_den.dat
/*The following is the source definition part.*/
/* The number of particles to be simulated. Invalid if NOT using phase space file (PSF), but needed to be listed.*/
simulation history:
1000000
use phase space file as source (0 for no, 1 for yes):
0
source file:
input/pointsource.txt
/*Particle type in PSF. Invalid if NOT using phase space file.*/
particle type of phase-space file for source (0 for positron, 1 for photon):
0
positron range consideration (0 for no, 1 for yes):
0
/* Invalid if using phase space file. In this case, just keep the two lines as default. */
start time, end time for the radioactive decay process (in seconds):
0 120
/*Record photons after its escaping from the phantom. The PSF recording surface must be between the phantom and the detector.*/
center (x, y, z) and radius for the photon PSF recording surface in the global coordinate (in cm):
0 0 0 5
/* Cutoff energy for photon transport in the phantom and detector.*/
photon absorption energy (in eV):
1e3
/* The following are the detector and digitizer parts. */
detector geometry file:
input/config8.geo
/* Quadratic surface definition for irregular shape detector configuration. For regular detector, just leave it as default. */
total number of quadratic surfaces and the TEN parameters (x2 y2 z2 xy xz yz x y z constant) to define each quadratic surface in the local coordinate (in cm):
1
0 0 0 0 0 0 0 0 0 1
/*Readout depth from 1 to 3 means panel, module, crystal.
policy 0 means winner take all  only select the event with maximum deposited energy if multi events locate in the same depth; policy 1 means centroid by energy average. Positions will be averaged among multi events if they locate in the same depth according to their deposited energy.
If policy is 1, readout depth will be set to 2 automatically. */
readout depth (1 to 3), policy (1 and 2):
2 1
/* Energy cutoff caused by electronic device limitation. */
thresholder energy before deadtime (eV):
50000
/*Energy and spatial blurring is realized via Gaussian blurring, with the resolution R defined as the FWHM.
Energy blurring 
-	policy 0, inverse square law, , 
-	policy 1, linear law, .
Spatial blurring follows universal Gaussian blurring for all three dimensions.*/
policy, reference energy (in eV), reference resolution (in eV) and slope for energy blurring, and resolution for space blurring (in cm):
1 662000 0.05 0 0
/*Deadtime part contains deadtime level, deadtime policy and dead time length.
Deadtime level generally follows the readout depth. As for the deadtime policy, 0 for paralyzable and 1 for nonparalyzable. The differences between the two policies can refer to GATE manual. */
deadtime level (1, 2, 3), deadtime policy (0 for paralyzable, 1 for nonparalyzable), and deadtime duration (in micro-second):
3 0 2.2
/* User defined energy window to form singles. */
thresholder and upholder for energy window (in eV):
30000 700000
```

