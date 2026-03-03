# When-Patterns-have-no-Meaning

In the age of machine learning, coupled with the explosion of chemo-biological data from MD simulations, the goal is to leverage these to understand key drivers of changes, binding, catalysis at the macromolecular level.

*In the study by [Brownless et. al. 2025;](https://doi.org/10.1021/acs.jpcb.4c08824) machine learning was used to characterize the most important residues that drive interactions between two corona virus variants (SARS-CoV and SARS-CoV2) and their biological target (ACE2 receptor).*

Even though the authors strongly believes that such technique could be extended to other possible protein-receptor interactions, **I decided to try it out on Ligand-protein interactions.**

What key residues are responsible for binding/conformational changes upon ligand binding--- that can be used to differentiate bound vs unbound conformations.

The data used here is from a malaria study **(unpublished)**, where we performed 400ns simulations of the apo and MMVMMV019313-bound bifunctional farnesyl/geranylgeranyl pyrophosphate synthase (FPPS/GGPPS) from *Plasmodium falciparum*. 

**Here is the hypothesis:**

*Fluctuations and changes in the position of residues around the active can model the differences between a bound and unbound conformation. This fluctuations can be tracked by the **per-atom distances** between all possible residue-atom combinations of residues withing a cut-off distance from the ligand around the active site*

To get the data, I defined the set of all **residues within 8Å** from the ligand:

```text
RES_INPUT="63-73,97-98,100-112,117-118,138,169-170,172-178,180-181,218,221-235,258,261-273,287-292,366-367,369-378"
```

And the set of all heavy atoms that may interact with each other:

```text
ATOMS=ATOMS <<< "N,CA,C,O,CB,CG,CD,CE,CZ,NE,NE1,NE2,NZ,OE1,OE2,OD1,OD2,SG,SD,CG1,CG2,CD1,CD2,ND1,ND2,NH1,NH2,OG,OG1,OH,CH2"

```

Then I sought to generate unique pairs of each of the residues and the atoms using this [bash file](Link). Next was to generate a list of distance between pairs, for cpptraj, of all the residue-atom lists in a unidirectional way (A→B same as B→A) into *pair_generated.in* file. A crosssection looks like this:

```text
distance d_63_CD_100_CG1 :63@CD :100@CG1 out d_63_CD_100_CG1.dat
distance d_63_CD_100_CG2 :63@CD :100@CG2 out d_63_CD_100_CG2.dat
distance d_63_CD_100_CD1 :63@CD :100@CD1 out d_63_CD_100_CD1.dat
.....
```
This was used as the input file for the cpptraj command of ambertools. Due to the massive size of this file (containing a possible 1.8 million distances), it was feed into cpptraj at 5000 lines per time (corresponding to 300+ cycles) using this [script](Link). These data were merged and converted into csv files.

**Concerning our trajectory**; 400ns simulation corresponds to 200000 frames at NTWRX=1000. So, we parsed it at a stride of 20 to give a total of 10,000 frames (or data points)

To further reduce the size of the dataset, every 10th datapoint was sampled from each of the bound and unbound datasets and pooled together to give data consisting of 2000 points only. Since most of the atom-pairs are not present in some residues, we came down to 200K+ possible distances (columns or descriptors). Also, we selected only CA and CB distances between residues, thereby reducing our column length down to  from 1.8 million. To combine this dataset, we defined a new column (State) with categorical variables of "bound" and "unbound" to depict the apo and MMV-bound states. 

Then this dataset was used as input in our machine learning study. Most of the codes and methodology we will apply below are simple aplication sof codes used by [Brownless et. al. 2025;](https://doi.org/10.1021/acs.jpcb.4c08824)


Load modules and read in the dataset
```python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Your MXMD/clustering results
df = pd.read_csv('/path-to-your-file/Full_dataset_Apo_MMV.csv')
```

As already mentioned, to reduce the size of the dataset, we selected only CA and CB distances. Also, to handle missing values (empty cells), we filled the "NAN" with 50Å. The rational for this is that such large distances are impossible (since max would be 16-20Å) and depict that no such interactions exist. This choice will also be clear later when we adopt the concept of **Residue Importance** used in the main work.

```python
#Select only columns with CA and or CB distances

import re
pattern = re.compile(r"^d_.*_(CA|CB)_.*_(CA|CB)$")
cols = [col for col in df.columns if pattern.match(col)]
cols.append("State")

filtered_df = df[cols]
# then we will fill the NAN with the max in each column (since it depicts non interacting); around 50A
df_full = filtered_df.fillna(50)
df_full.isna().sum().sum()
df_full.isna().sum().sort_values(ascending=False)
```

To make progress, we removed the last column (STATE) and normalized the dataset using Z-score

```python
# Separate features and labels
df_features = df_full.iloc[:, 1:-1]  # Middle columns = residue features
df_labels = df_full.iloc[:, -1]      # Last column = Cluster/State labels

# Process features only (inverse + z-score) + normalization
df_features_numeric = df_features.select_dtypes(include=['float64'])
df_importance = 1 / df_features_numeric
df_scaled = (df_importance - df_importance.mean(axis=0)) / df_importance.std(axis=0)
```
