# Project 4: Scene Recognition with Bag of Words

## Setup
We will be installing a **NEW** environment for this project; follow the instructions below to set up the env. If you run into import module errors, try “pip install -e .” again, and if that still doesn’t work, you may have to create a fresh environment.

1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyway.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj4_env_<OS>.yml`
3. This should create an environment named 'cs4476_proj4'. Activate it using the Windows command, `activate cs4476_proj4` or the MacOS / Linux command, `conda activate cs4476_proj4`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj4_code/proj4.ipynb`
6. Ensure that all sanity checks are passing by running `pytest` either inside the "proj5_unit_tests/" folder, or directly in the project directory.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` and submit to Gradescope (don't forget to submit your report to Gradescope!).
