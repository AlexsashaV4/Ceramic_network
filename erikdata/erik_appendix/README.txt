Project Title:
Pathfinder


Project Description:
Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. For a broader overview: see Kroon (in prep.)

The basis is a network representation of the ceramic chaîne
opératoire based on the overview of ceramic production
techniques by Roux (2019). 

A ceramic production process is a combination of specific 
techniques. The network representation captures all possible 
combinations of ceramic production techniques. Therefore, all 
possible production processes exist as paths through this 
network. 

Pathfinder treats this network as a probability space, in which 
each path is a series of choices based on the syntax, previously 
applied techniques, hydric state of the vessel, and techniques 
known to the potter. This conceptualisation enables three 
operations:

1) Random generation of paths. In this case an algorithm 
generates a probable ceramic production process by starting at 
a given node (wet clay), randomly selecting one neighbour, 
appending it to the list, and reiterating this process until 
the path satisfies criteria for a completed ceramic production 
process (see Kroon in prep. Section 4.2 for a complete description).

2) Guided random generation of paths. Similar to the algorithm above,
but the chance of selecting a neighbouring node is proportional to 
the number of times that combination of techniques exists in a prior
dataset. This mode enables working with highly fragmented or legacy
datasets (see Kroon in prep. Section 4.2, Appendix F).

3) Comparison of sets of paths. Each path (or ceramic production 
process) consists of a number of choices to combine specific 
production techniques. Pathfinder compares these choices in two bodies 
of vessels by looking at all the edge weights in the network at the 
same time. These edge weight distributions are compared using the 
Wasserstein distance. The script automatically generates a third edge 
weight distribution (based on 1000 randomly generated paths) as a 
control group (see Kroon in prep. Ch. 4).

Pathfinder makes use of .csv files in utf-8 format (separator = ';') 
for data input and output. This limits the number of paths which can 
be generated in one go to ca. 100k. 

Pathfinder generally works best with complete paths. However, fragmented 
data can be used in to generate probable paths using guided random path
generation (see Kroon in prep.).

Pathfinder uses the choice() and count() methods at crucial points
in the process. These methods can slow computation when working with
large datasets.


How to Use:
Pathfinder is a Python script and uses several open access 
libraries and packages in Python (v3.10.2): 
	- networkx (2.6.3) (Hagberg et al. 2008) 
	- scipy (1.9.1) (Virtanen et al. 2020)
	- random 
	- csv
	- collections

Apart from the script, Pathfinder also comes with a number of 
supporting files in the folder 'ceramics'. These files are vital
to the algorithm and must be left unmodified in the folder. These
files are:

- bok_1.csv
- bok_2.csv
- link_list.csv
- nodes_list.csv
- output_comparisons.csv
- output_random_paths.csv
- output_transl.csv
- seeded_output.csv

Pathfinder may output data to some of these files. In which case
you can copy the data from these files to another file. Empty these
files between iterations. The interface regularly prompts the user
to do so.

After installing Python and the relevant libraries, download all
files (script and support files) in the folder 'Code'. Then open 
command prompt and navigate to pathfinder.py. This opens the 
interface, from where you can generate and compare ceramic production 
processes.

If you want to use Pathfinder for your own datasets, make sure to format
the data properly. The comments in the script for the interface explain
the formatting guidelines (see also Kroon in prep.). Here are six basic
guidelines:

1) Make a .csv file. Each row in the file is a path, or production process
from start to end.

2) Each cell in a row must be filled with a string which matches exactly(!)
those found in the column 'name' in the file nodes_list.csv. No empty cells
may appear in the middle of a path.

3) The first cell on a path must always be 'Wet clay'.

4) Each node in the path (i.e. cell) must have an incoming link from the 
node in the previous cell. See link_list.csv for an overview of links.
 
5) No node name may feature twice in the same path (i.e. the path may 
not loop).

6) All nodes have a tolerance. (W = wet, L = leather-hard, D = dry, F = 
fired). Nodes may only appear on positions in the path which conform their
tolerance. e.g. coiling has tolerance 'W' and must therefore appear between
the nodes 'wet clay' and 'leather-hard; burnishing has tolerances 'L' and 
'D', meaning it must appear after the node 'leather-hard', but before the node 
'firing'.

Credits:
Work on this project was supported by the Dutch Research Council 
(NWO) project "What's in a pot? Transformations during the Third 
Millennium BC from the Perspective of Ceramic Technology" (PGW.
18.003/6330).


License:
The script is provided as is under a CC-BY 4.0 license.


Sources:
Hagberg, A.A., D.A. Schult and P.J. Swart (2008). Exploring network structure,  
dynamics, and functioning using NetworkX. In: G.Varoquaux, T. Vaught and J. 
Millman (eds.), Proceedings of the 7th Python in Science Conference. SciPy 
2008: Pasadena (CA), pp. 11-5. 

Kroon, E.J. (in prep.). Serial Learners: Interactions between 
Funnel Beaker West and Corded Ware Communities in the Netherlands 
during the Third Millennium BCE from the Perspective of Ceramic 
Technology. PhD Dissertation. Leiden University, Faculty of 
Archaeology.

Roux, V. (2019). Ceramics and Society: A Technological Approach to
Archaeological Assemblages. Cham: Springer Nature.

Virtanen, P., R. Gommers, T.E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, 
E. Burovski, P. Peterson, W. Weckesser, J. Bright, S.J. van der Walt, M. Brett, 
J. Wilson, K.J. Millman, N. Mayorov, A.R.J. Nelson, E. Jones, R. Kern, E. Larson, 
C.J. Carey, İ. Polat, Y. Feng, E.W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, 
R. Cimrman, I. Henriksen, E.A. Quintero, C.R. Harris, A.M. Archibald, A.H. Ribeiro, 
F. Pedregosa, P. van Mulbregt, and SciPy 1.0 Contributors (2020). SciPy 1.0: 
Fundamental algorithms for scientific computing in Python. Nature Methods 17(3), 
261-72.