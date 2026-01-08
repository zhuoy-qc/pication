🧬 Pi-cation-Dock 
Our code applys to any protein pocket contains at least one postively charge amino acid. This repository provides source code for our RingDock pipepline, focusing on pi-cation interactions (cations from protein interacts with ligand aromatic rings). Other non-colvaent interactions developing...

![My Diagram](RingDock_workflow.png)



⚙️ Environment Setup 
We recommend using Conda to set up the environment.

conda env create -f ringdock_pi-cation_env.yml

conda activate ringdock_pi-cation_env

📁 Preparation 
Download the ring_sdf_files dir and set the path in the codes to match this dir path


📁 Dataset-Based Usage (e.g., PoseBuster)
To run the full pipeline on a dataset like PoseBuster:

Download the dataset of interest.cd to the directory containing all PDB ID dirs.RUN move.py to move the dir containing PI-CATION interaction to "with_pication" dir. 
cd to "with_pication" dir


Run the following scripts in that directory:

python pi-cation-analysis.py, which finds all pi-cation interactions and list the distance,offset,Rz of these interactions.
python 1_sampling.py

NOTE THAT when running the sampling.py, you can change a few lines, namely scoring functional, supporting : vinardo, vina, default.


python 2_model.py




🧪 Single PDB Usage
You can also analyze individual PDB files using the utilities provided:

Navigate to the utils/ directory.

We provide examples on several PDB files, and provide codes to reproduce our paper results.


📊 Output & Evaluation
The analysis codes include:

Compute Recovery rate for π-cation interactions

Structural error metrics


