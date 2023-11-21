# Reinforcement Learning in a Spiking Neural Model of Striatum Plasticity

This repository hosts the simulation code and related materials for the scientific article:

> González-Redondo, Á., Garrido, J., Arrabal, F. N., Kotaleski, J. H., Grillner, S., & Ros, E. (2023). Reinforcement learning in a spiking neural model of striatum plasticity. Neurocomputing, 548, 126377.

The primary focus is on a spiking neural model that implements reinforcement learning mechanisms in the context of striatum plasticity.

Corresponding author and maintainer: Álvaro González-Redondo (alvarogr@ugr.es).


## Compilation instructions

1. Navigate to the `edlut_lib/compiled` directory:
   - Use the command: `cd edlut_lib/compiled`

2. Run CMake to configure the project and prepare the build process:
   - Use the command: `cmake .. -Dwith-python=3`
   - Ensure you have CMake installed and Python 3 is available on your system.

3. Compile the project:
   - Use the command: `make`
   - Make sure you have the necessary compilers and dependencies installed.

4. Return to the root folder of the project:
   - Use the command: `cd ../..`

5. Create a symbolic link to the compiled Python library in the root folder:
   - Use the command: `ln -s edlut_lib/compiled/python/ edlut`
   - This step is essential for Python to find the compiled modules.

6. Run the Jupyter Notebook:
   - Make sure Jupyter Notebook is installed.
   - Open the notebook `01 - Network generalized single run.ipynb` using Jupyter Notebook.
   - Execute the notebook cells to run your code.
