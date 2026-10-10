"""
Author: E.J. Kroon

v1 June 27, 2023

Please refer to Kroon (in prep.) when using this program.

Pathfinder is meant to compare, generate, and simulate ceramic
chaînes opératoires. This is the interface. It calls up the other 
modules as needed.

This is the interface. From the interface, you can call the modules.
There are also several support files in the folder ceramics.

All input and output files are .csv files set to utf-8, separator=';''.
Cut or copy generated date from the indicated files when done, but do 
not change file structure or names to avoid errors.

The interface enables users to generate various data. Always delete 
temporary date between runs, especially when comparing paths. 
Otherwise the data files will contain (and compare) data from previous 
runs as well as the current run.

This script makes use of separate open source library in python:
csv. See the documentation of said libary. The script is provided
as is under a CC-BY 4.0 license.

"""

import module_random_path_generator as r_g
import module_converter_tool as converter
import module_compare_boks as compare
import module_p_guided_random as psim
import module_l_guided_random as lsim
import csv

#Function to check if user wants to quit.
def check_for_quit(a):
	if a == 'quit':
		print("\nExiting Pathfinder.")
		exit()
	else:
		return a

#Function to check if user wants to delete temporary variables
def del_temp_vars(*f):
	t_a = ('yes', 'no')
	print('Please cut any data you need from the file. Do not change file names or structure.')
	print('Deleting temporary data is recommended in between runs.')
	print('Extract any data you need beforehand.')
	while True:
		temp_data = input("Would you like to delete temporary data? yes/no ")
		check_for_quit(temp_data)
		if temp_data in t_a:
			if temp_data == 'yes':
				for temp_file in f:
					t = open(temp_file, "w")
					t.truncate()
					t.close()
				print("Temporary data deleted.")
				break
			elif temp_data == 'no':
				print('Temporary data kept.')
				return
		else: 
			print("Statement not recognised. Try one of the stated options:")
			print(f'{t_a}')

#Greeting text
print("\nWelcome to Pathfinder!")
print("This program is meant for network analysis of ceramic chaînes opératoires.")
print("Please refer to Kroon (in prep.) when using this software.")
print("Enter 'quit' anytime to exit pathfiner.")
print("\nLet's get started.")

#Define program state
state = 'set-up'

#Menu for selecting mode
while state == 'set-up':
	unrecognised_statement = "\nYour input was not recognised. Try one of these options:"
	#Ask input for mode
	p_a = ('1', '2', '3')
	message = ("\nWould you like to:")
	message += (f"\n\t1) Generate random paths")
	message += (f"\n\t2) Compare two assemblages")
	message += (f"\n\t3) Use guided random path generation")
	message += (f"\n ")
	mode = input(message)
	check_for_quit(mode)
	
	#Call relevant module based on user input
	if mode in p_a:
		if mode == '1':
			while True:
				#Data is dumped into an .csv file, so upper bound is ca. 100k paths
				n = input("How many random paths would you like to generate? ")
				check_for_quit(n)
				try:
					number = int(n)
					break
				except ValueError:
					print("Please enter an integer.")
			state = 'random generation'

		elif mode == '2':
			state = 'compare paths'

		elif mode == '3':
			state = 'guided random generation'

	#For invalid input
	else: 
		print(unrecognised_statement)
		print(f'{p_a}')

#Mode for random generation of paths
while state == 'random generation':
	# Call module and generate user-specified number of paths
	r_g.random_path_generator(number)

	#Exit message
	e_m = "\nAll done!"
	e_m += "\nFind the output in the folder pathfinder/ceramics/output_random_paths.csv."
	print(e_m)
	del_temp_vars('ceramics/output_random_paths.csv')
	print(f'Exiting pathfinder.')
	break

#Mode for comparing bodies of specific chaînes opératoires
while state == 'compare paths':
	#Instructions
	message = "Place two .csv-files with paths in the data folder. Format the data rigidly"
	message += "\nfollowing the guidelines. Failure to do so may result in errors."
	print(message)
	
	"""
	The files should be .csv in utf-8 with separator = ';'. Each row must be a complete
	specific chaîne opératoire starting with wet clay. Each cell in the row must be 
	a str from exactly those listed in ceramics/nodes_list.csv in the column 'Name'. The 
	code assumes the first empty cell in a row is the end of the path. Paths must not loop. 
	Each cell except the first must have an incoming edge towards it from the previous cell
	in the file links_list.

	e.g. 
	'Wet clay';'Modelling';'Coiling';...
	'Wet clay';'Modelling';'Pressure';...

	For examples, see also: ceramics/FBW.csv, MI.csv, EFG.csv,LFG.csv, CW.csv.

	Incorrect formatting may result in unexpected/undetected errors, because the
	converter module cannot connect the node Name to a node ID in nodes_list, or the 
	compare module cannot find a link to attribute a value to.

	Body of knowledge A (BoK A) is the base of the comparison. BoK B is compared to it.
	The module automatically generates 1,000 specific chaînes opératoires for BoK C and
	compares these to BoK A (cf. Kroon in prep. Tab. 4.1).

	Keep in mind that processing time increases with the number of paths and number of 
	nodes in a path.

	If user quits having entered one BoK, the temporary data is automatically deleted
	to prevent it from affecting the next comparison. 
	
	"""

	#Prompt user for file paths to BoK A and B, and call converter module
	BoK_A = input("Specify the file path for Body of knowledge A ")
	check_for_quit(BoK_A)
	try:
		converter.converter_tool(BoK_A, 'ceramics/bok_1.csv')
		print("File conversion is done.")
	except FileNotFoundError:
		print("File path is invalid, please try again.")
		exit()
	
	BoK_B = input("Specify the file path for assemblage B ")
	if BoK_B =='quit':
		d = open('ceramics/bok_1.csv', "w")
		d.truncate()
		d.close()
		print("Temporary data deleted.")
	else:
		try:
			converter.converter_tool(BoK_B, 'ceramics/bok_2.csv')
			print("File conversion is done")
		except FileNotFoundError:
			print("File path is invalid, please try again.")
			#Zorg dat je bij een error ook data verwijdert

	#Call compare module, and feed converted files
	print("Comparing assemblages.")
	compare.compare_boks('ceramics/bok_1.csv', 'ceramics/bok_2.csv')
	#Exiting message
	print(f"\nAnalysis done.")
	del_temp_vars(
				'ceramics/bok_1.csv', 'ceramics/bok_2.csv', 
				'ceramics/output_random_paths.csv',
				'ceramics/output_comparisons.csv')	

#Mode for guided random generation of paths or simulation
while state == 'guided random generation':
	message = "Place a.csv-file with a link list or paths in the data folder for the random seed."
	message += "\nPlease format the data rigidly according to the guidelines."
	message += "\nOtherwise unexpected errors may result."
	message += "\n\nLet's set the variables."

	"""
	The algorithm for guided random path generation can generate paths based
	on an existing dataset or random seed (cf. Kroon in prep. Section 4.2). 
	The data can be fed to the algorithm in two ways: 1) as a link list or 2) 
	as a set of paths. The link list is used to plot a directed graph with 
	weighted edges. The set of paths is first converted into a link list. 

	Format the paths accordingto the guidelines for comparing bodies of
	knowledge(see above). 

	A link list should be a csv (utf-8, sep = ';') with the headers source,
	and weight. Weight should be an integer corresponding to the number or
	percentage of paths which runs along a link. The nodes on the link list 
	should be identical to those on the nodes list. For an example, see 
	ceramics/link_list.csv.  

	e.g. 

	'Source';'Target';'Weight'
	1;2;1
	1;3;1
	...

	Replacing the values in the column 'weight' in
	this list also works, but make sure to copy the file and re-name it 
	before making these changes.

	Working with a link list requires the user to input information on the
	number of original vessels (or 100 if percentages are used as edge 
	weights). The algorithm uses this value to determine the number of 
	vessels which underwent	operations after firing. This information is 
	used to prevent the algorithm from looping through all post-firing 
	treatments. The value is automatically extracted from a random seed 
	with paths.

	The algorithm needs four further arguments:

	1) Default value for the undetected links: An integer which is attributed
	to all links with weight 0 in the link list or all non-detected links
	in the set of paths. Combine with a multiplier to introduce 'random
	drift' in path generation or set to 0 to stick with links found in 
	base dataset.

	2) Multiplier for detected links: The edge weights in the link
	list (or those resulting from conversion of a set of paths) will be 
	multiplied by this integer. Together with the default value, this 
	variable can be used to introduce 'random drift' in a simulation.
	Set to 1 to switch it off.

	3) Recursion factor: The algorithm for guided random path generation
	breaks up each generated path after storing it, and increments the 
	weight of the edges in the path by 1. This value is multiplied
	by the recursion factor beforehand. Set the recursion value to 0 to 
	swith off recursion. Recursion does not affect the link list or paths 
	fed into the algorithm. The value should be an integer. The recursion
	factor can be used to for simulating longitudinal patterns.

	4) Number of interations: This is the number of times the algorithm 
	generates a paths (and increments the edge weights). This value should
	be an integer. Keep in mind the algorithm parses a list of target node IDs 
	where each node ID is included as many times as the weight of the edge
	between source and target. Therefore, a large number of iterations
	(combined with high edge weights) significantly increases computation time.

	If the paths from guided random generation are intended for a comparison,
	make sure to put them through conversion first. 
	
	"""
	print(message)
	#Prompt user for variables.
	while True:
		s_f = input("Specify the file path to the random seed. ")
		check_for_quit(s_f)
		try:
			with open(s_f, 'r') as f:
				reader = csv.reader(f)
				break
		except:
			FileNotFoundError
			print("File not found. Specify file path.")

	while True:
		d_v = input("Specify the default value for non-detected links. ")
		check_for_quit(d_v)
		try:
			int(d_v)
			break
		except:
			ValueError
			print('Please enter an integer.')

	while True:
		l_m = input("Specify the multiplier for detected links. ")
		check_for_quit(l_m)
		try:
			int(l_m)
			break
		except:
			ValueError
			print('Please enter an integer.')

	while True:
		r_f = input("Specify the recursion factor. ")
		check_for_quit(r_f)
		try:
			int(r_f)
			break
		except:
			ValueError
			print('Please enter an integer.')

	while True:
		i = input("Specify the number of iterations. ")
		check_for_quit(i)
		try:
			int(i)
			break
		except:
			ValueError
			print('Please enter an integer.')

	while True:
		m = input("Is the random seed a (1) link list or (2) set of paths? 1/2 ")
		check_for_quit(m)
		if m == '1':
			o = input("Enter the number (integer) of paths which undergo operations after firing: ")
			lsim.guided_random_generator(s_f, d_v, l_m, r_f, i,o)
			break
		if m == '2':
			converter.converter_tool(s_f, 'ceramics/output_transl.csv')
			print("File conversion is done.")
			psim.guided_random_generator('ceramics/output_transl.csv', d_v, l_m, r_f, i)
			break
		else:
			print("Please enter 1 or 2.")

	#Exit message
	e_m = "\nAll done!"
	e_m += "\nFind the output in ceramics/seeded_output.csv."
	e_m += "\nPlease cut the requested information from these files"
	e_m += "\nbut do not change the file structure or name."
	print(e_m)
	del_temp_vars('ceramics/seeded_output.csv', 'ceramics/output_transl.csv')
	print(f'Exiting pathfinder.')
	break
