# LatticeAnalyser

This a program for analysis data created from [GLAC](https://github.com/hmvege/GLAC), a Lattice QCD program for generating and flowing quenched configurations. This program was used for a M.Sc. thesis located [here](https://github.com/hmvege/LQCDMasterThesis).

---

#### Package contents:
| Folder | Description |
| ------ | ----------- |
| `analysis_batches` | Contains analysis of different batches. These are the ones that will be called by `LQCDAnalyser.py`, the main anlaysis program. |
| `pre_analysis` | Contains program for the base analysis of the data. |
| `post_analysis` | Files for performing the different line fits, continuum extrapolations and general gathering of plots into common windows. |
| `statistics` | Contains different statistical tools, such as bootstrap, jackknife, time series bootstrap, autocorrelation and tools to be used in the parallel methods of these. |
| `tools` | Contains files for reading and writing data. |

