import sys
import re

C = '.circuit'			#stating the format	
E = '.end'				#start and end to compare with the lines read

print("The netlist file to be read: {}".format(sys.argv[1]))

start = -1
end = -1


try:
	with open(sys.argv[1]) as f:
		lines = f.readlines()
		for l in lines:
			if C == l[:len(C)]:				#Identifying the start of the circuit defn
				start = lines.index(l)
			elif E == l[:len(E)]:			#Identifying the end of circuit defn
				end = lines.index(l)
				break 

		if start>=end :
			print("Invalid circuit definition")
			exit(0)
		
		for line in reversed([' '.join(reversed(line.split('#')[0].split()))			#Identifying the text excluding the comments
			for line in lines[start+1:end]]):
				print "\n"
				print(line)												#Printing the reversed line
				info = line.split()										#Storing the line as a list
				ele = "".join(re.split("[^a-zA-Z]*", info[-1]))			#Identifying the Alphabet to know the type of element
				if ele == 'R':
					print("\nPassive element : Resistor\nBetween Nodes : {} and {} and Value : {}".format(info[1],info[2],info[0]))			#printing info of the element that is described in the line
				elif ele == 'C':
					print("\nPassive element : Capacitor\nBetween Nodes : {} and {} and Value : {}".format(info[1],info[2],info[0]))
				elif ele == 'L':
					print("\nPassive element : Inductor\nBetween Nodes : {} and {} and Value : {}".format(info[1],info[2],info[0]))
				elif ele == 'V':
					print("\nIndependent Voltage Source\nBetween Nodes : {} and {} and Value : {}".format(info[1],info[2],info[0]))
				elif ele == 'I':
					print("\nIndependent Current Source\nBetween Nodes : {} and {} and Value : {}".format(info[1],info[2],info[0]))
				elif ele == 'E':
					print(" Dependent Source : VCVS")
				elif ele == 'G':
					print(" Dependent Source : VCCS")
				elif ele == 'H':
					print(" Dependent Source : CCVS")
				elif ele == 'F':
					print(" Dependent Source : CCCS")
				else: pass
	f.close()
except IOError:
	print('Invalid file')
	exit()
