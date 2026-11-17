# Inferring Temporal Epidemiological Dynamics with Deep Learning using simulated phylogenetic trees.

This code takes a simulated tree and estimates 8 numbers from it. These are the birth rate and the death rate in each of 3 time periods and 2 times when the rates change. The model used is an attention based MLP that reads the tree as a list of events.

------------------------------------------------------------------------

## 📋 What you need

-   A Mac, a Linux computer, or a Windows computer (Windows needs the free Ubuntu app, see Part 1).
-   A Google account, for Google Colab. Training needs a GPU, and Colab gives you one.

### Part 1. Set up

Step 1. Open a terminal.

-   **Mac:** press Command and Space, type `Terminal` and press Enter.
-   **Linux:** open the Terminal app.
-   **Windows:** open PowerShell as administrator, type `wsl --install` and restart the computer after installation is finished. Then open the **Ubuntu** app from the Start menu and type all commands there.

Step 2. Download the code and go into its folder:

```         
git clone https://github.com/vinesh4/temporal-epidemiological-dynamics-dl.git
cd temporal-epidemiological-dynamics-dl
```

If git is missing, a Mac asks you to install it, and on Ubuntu you type `sudo apt install git`.

Step 3. Make a virtual environment, turn it on and install the packages. You need Python 3.10 or higher (check with `python3 --version`). On Ubuntu, first type `sudo apt install python3-venv python3-pip`.

```         
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Part 2. Make the data

Step 1. Do a quick test. It makes 20 small trees per set and finishes in less than a minute:

```         
python simulation.py 69 --n-free 20 --n-different 20 --n-leaves 50 --tag test
```

Step 2. Run the full simulation (takes around 2-4 hours depending on how many CPU cores your system has. The more CPU cores you have, the less time it takes to run the simulations):

```         
python simulation.py
```

The simulation uses all the CPU cores of your computer by default. To use fewer, run ```python simulation.py --n-jobs 4``` instead..

Step 3. When it finishes, the folder `outputs` holds two zip files. Their names start with `3-Interval-ChangeTime-free-` and `3-Interval-ChangeTime-different-`.

#### Or make the data in Google Colab

Colab gives only 2 CPU cores, so this takes very long (longer than 8 hours), and the free version can disconnect before it finishes. Use it only if you have no strong computer.

Step 1. Go to <https://colab.research.google.com> and click **File**, then **New notebook**.

Step 2. Type this in the first cell and run:

```         
!git clone https://github.com/vinesh4/temporal-epidemiological-dynamics-dl.git
%cd temporal-epidemiological-dynamics-dl
!pip install -r requirements.txt
```

Step 3. In a new cell, run the quick test (less than a minute):

```         
!python simulation.py 69 --n-free 20 --n-different 20 --n-leaves 50 --tag test
```

Step 4. In a new cell, run the full simulation and keep the tab open until it finishes:

```         
!python simulation.py
```

Step 5. Click the folder icon on the left, open `temporal-epidemiological-dynamics-dl`, then `outputs`. Click the three dots next to each of the two zip files and choose **Download**.

### Part 3. Train and score the model

Step 1. Go to <https://colab.research.google.com>, sign in, click **File**, then Open notebook, click the GitHub tab, paste <https://github.com/vinesh4/temporal-epidemiological-dynamics-dl> and choose main.ipynb.

Step 2. Click **Runtime**, then **Change runtime type**, choose **L4 GPU if you have Colab Pro** (a T4 GPU is free and also works but is slower) and click **Save**.

Step 3. Click the folder icon on the left and drag the two zip files from the `outputs` folder into the panel. Do not unzip them.

On Windows, you can find the files from Ubuntu in File Explorer by typing `\\wsl$\Ubuntu` in the address bar.

Step 4. Click **Runtime**, then **Run all**, and do not change any settings. Training takes about 2.7 hours on L4 GPU and longer on T4, so keep the tab open. The overall error score should be about 0.1466. Differences of about 0.005 are normal on a GPU. The result files appear in the folder panel. To save, click the three dots next to each and choose **Download**.

### Part 4. Test the models on real trees

`real_data_test.ipynb` applies the final MLP (and the CNN for comparison) to three real trees (HIV in Zurich, influenza H3N2, SARS-CoV-2) and checks every step of the tree processing.

Step 1. Open `real_data_test.ipynb` in Colab from GitHub (as in Part 3) and choose a GPU.

Step 2. Drag in the same two zip files as in Part 3 (each Colab notebook has its own session).

Step 3. Download the influenza tree from <https://nextstrain.org/seasonal-flu/h3n2/ha/2y> (**Download data** → **Newick (timetree)**), rename it to `nextstrain_seasonal-flu_h3n2_ha_2y_timetree.nwk` and drag it in. It is built from GISAID data, so we cannot share it, and Nextstrain updates it, so influenza results may differ slightly from ours. The other trees and the model weights are downloaded automatically.

Step 4. Keep `RETRAIN = False` and click **Runtime** → **Run all** (about 15 minutes). Allow Google Drive access, or set `USE_DRIVE = False`.

> Note: the notebook contains its own training code because `main.ipynb` does not save weights; the weights trained with this code are downloaded from Google Drive, so no training runs.

Step 5. Check the results:
- MLP seed scores about 0.1507, 0.1505 and 0.1475 (overall about 0.146; it differs slightly from the 0.1466 of Part 3 because the weights come from a separate run).
- Section 10 should show 34 PASS and 2 NOTE, and no CHECK.
- Download `m4_results.zip` from the folder panel. It holds the tables, plots, checks and the files for an optional visual check in iTOL (steps in Section 10 of the notebook).

> Do not share the influenza files in `m4_results.zip`: they come from GISAID data.
