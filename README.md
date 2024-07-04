A set of scripts for parameterising equivalent circuit models using PyBOP. Note that these are very much rough research scripts, rather than polished and carefully designed software!

Structure is as follows...
- `datareaders.py` contains functions for loading in battery cycler data from Neware and BaSyTec cyclers, splitting into GITT pulses, and extracting OCV data
- `fitter.py` defines the model, generates fitting parameters for PyBOP, and runs the optimisation for a given list of GITT pulses; also handles helper tasks like coulomb counting
- `run_parameteriser.py` runs the parameterisation routines on a set of data files; it's set up for the data and file structures I'm using, but should serve as an example of how to put everything together
