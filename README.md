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

This part runs `real_data_test.ipynb`. It applies the final attention MLP, and the CNN for comparison, to three real trees: HIV in Zurich, influenza H3N2 and SARS-CoV-2. It turns each real tree into trees of 500 tips, estimates the 8 numbers, and converts the rates to "per year" and the change times to "years before the latest sample". At the end it checks every step of the tree processing.

Step 1. Go to <https://colab.research.google.com>, click **File**, then **Open notebook**, click the **GitHub** tab, paste <https://github.com/vinesh4/temporal-epidemiological-dynamics-dl> and choose `real_data_test.ipynb`.

Step 2. Click **Runtime**, then **Change runtime type**, choose a GPU (L4 or T4) and click **Save**.

Step 3. Click the folder icon on the left and drag in the same two zip files you used in Part 3. Do not unzip them.

> Note: this repeats Step 3 of Part 3 on purpose. Each Colab notebook runs in its own session, so the files from Part 3 are not there any more. The notebook needs the simulated trees to rebuild the same split and scaling as `main.ipynb` and to find the typical height of a simulated tree, which is used to put the real trees on the same time scale.

Step 4. Download the influenza tree yourself and drag it into the same panel:

1. Open <https://nextstrain.org/seasonal-flu/h3n2/ha/2y>.
2. Scroll to the bottom, click **Download data**, then **Newick (timetree)**.
3. Rename the file to `nextstrain_seasonal-flu_h3n2_ha_2y_timetree.nwk` if it has another name.

This tree is built from sequences shared through GISAID, whose terms do not allow us to share it, so it is not included. Nextstrain updates the tree regularly, so the influenza results will be slightly different from ours. The other two trees are downloaded automatically.

Step 5. Click **Runtime**, then **Run all**, and keep `RETRAIN = False`.

- Colab asks for access to your Google Drive. Allow it, because the results are saved there. If you prefer not to, set `USE_DRIVE = False` in the settings cell before you run.
- The notebook downloads the saved model weights and the Zurich and SARS-CoV-2 trees from Google Drive, so no training is needed. It takes about 15 minutes, mostly reading the simulated trees.

> Note: this notebook contains its own copy of the training code, which overlaps with Part 3. `main.ipynb` does not save its weights, so the models used on the real trees were trained in this notebook with the same code and settings, and their weights are shared on Google Drive. Training only runs if the weights cannot be downloaded or if you set `RETRAIN = True` (about 3 hours on an L4 GPU).

Step 6. Check the results.

- While loading, the notebook prints a score for each seed of the MLP: about 0.1507, 0.1505 and 0.1475. Their average prediction gives an overall error score of about 0.146.

> Note: this is close to, but not the same as, the 0.1466 of Part 3. The weights come from a separate training run with the same code, and differences of this size are normal on a GPU (see Part 3, Step 4).

- The main outputs are the table of estimates for each real tree (median and 95% range) and a box plot of the change times. For example, both models place the influenza changes about 6.6 and 4.6 to 4.8 years before the latest sample.
- Section 10 checks the tree processing and prints PASS, NOTE or CHECK for each check. Our run gave 34 PASS and 2 NOTE (real tips on zero-length branches are read as sampled ancestors). Any CHECK means a step did not work as expected.
- When the run finishes, `m4_results.zip` (tables, plots, the 500-tip trees, the checks and the iTOL files) appears in the folder panel. Click the three dots next to it and choose **Download**. With `USE_DRIVE = True`, the same files are also in `capstone_m4` in your Google Drive.

> The real trees have no known true values, so these numbers show whether the models give stable answers on real data, not how accurate they are.

Step 7 (optional). Look at the processed trees in iTOL.

1. Unzip `m4_results.zip` and open the folder `checks/itol`. For each real tree it holds the original tree (`_original.nwk`), the first processed tree in years with the original tip names (`_subtree1_years.nwk`) and the list of kept tips (`_kept_tip_names.txt`).
2. Open <https://itol.embl.de/upload.cgi> and upload `Zurich_HIV_original.nwk`. In the control panel on the right, choose **Rectangular**, set **Branch lengths** to **Use** and **Labels** to **Hide**.
3. In a new browser tab, upload `Zurich_HIV_subtree1_years.nwk` with the same settings. The two trees should look the same.
4. For SARS-CoV-2, open `Nextstrain_SARS_CoV_2_kept_tip_names.txt`, add the two lines `PRUNE` and `DATA` at the top, save it, and drag it onto the original tree in iTOL to keep only those tips. Then compare it with `Nextstrain_SARS_CoV_2_subtree1_years.nwk`. The only difference should be extra branches of length 0 where polytomies were split.

> Do not upload the influenza files to iTOL or anywhere else. They come from GISAID data, which may not be passed on to third parties.
