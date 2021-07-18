import sys
import re
import numpy as np
import cmath

C = '.circuit'
E = '.end'
A = '.ac'
RES = "R"
CAP = "C"
IND = "L"
IVS = "V"
ICS = "I"
VCVS = "E"
VCCS = "G"
CCVS = "H"
CCCS = "F"
PI = np.pi


class Resistor:
	def __init__(self,name,n1,n2,val):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.val = val


class Capacitor:
	def __init__(self,name,n1,n2,val):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.val = val


class Inductor:
	def __init__(self,name,n1,n2,val):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.val = val


class Voltage_source:
	def __init__(self,name,n1,n2,val,phase=0):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.val = val
		self.phase= float(phase)


class Current_source:
	def __init__(self,name,n1,n2,val,phase=0):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.val = val
		self.phase = float(phase)


class vcvs:
	def __init__(self,name,n1,n2,n3,n4,val):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.node3 = n3
		self.node4 = n4
		self.val = val


class vccs:
	def __init__(self,name,n1,n2,n3,n4,val):
		self.name = name
		self.node1 = n1
		self.node2 = n2
		self.node3 = n3
		self.node4 = n4
		self.val = val


class ccvs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.val = val
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName


class cccs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.val = val
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName


Components = {RES: [],CAP: [],IND: [],IVS: [], ICS: [], VCVS: [], VCCS: [], CCVS: [], CCCS: [] }

nodes = []

if len(sys.argv)!=2:
	sys.exit("File name not given")
print("The netlist file to be read: {}".format(sys.argv[1]))
if (not sys.argv[1].endswith(".netlist")):
	sys.exit("File type not for expected circuit definition")


start = -1
end = -1
circuitFreq = 1e-100
i=0
j=0
try:
	netlistFileLines = []
	circuitFile = sys.argv[1]
	with open (circuitFile, "r") as f:
		for line in f.readlines():
			netlistFileLines.append(line.split('#')[0].split('\n')[0])
			# Getting frequency, if any
			if(line[:3] == '.ac'):
				circuitFreq = float(line.split()[2])
			# Setting Angular Frequency w
			w = 2*PI*circuitFreq
			# Finding the location of the identifiers
		#print(netlistFileLines)
		try:
			identifier1 = netlistFileLines.index(C)
			identifier2 = netlistFileLines.index(E)
			circuitBody = netlistFileLines[identifier1+1:identifier2]
		except:
			sys.exit("Netlist not in correct format")
		#print(circuitBody)
		for line in circuitBody:
		# Extracting the data from the line
		

			info = line.split()
			if info[1] not in nodes:
				nodes.append(info[1])
			if info[2] not in nodes:
				nodes.append(info[2])


			if info[0][0] == RES:
				Components[RES].append(Resistor(info[0], info[1], info[2], float(info[3])))
				# Capacitor
			elif info[0][0] == CAP:
				Components[CAP].append(Capacitor(info[0], info[1], info[2], float(info[3])))
				# Inductor
			elif info[0][0] == IND:
				Components[IND].append(Inductor(info[0], info[1], info[2], float(info[3])))
				# Voltage Source
			elif info[0][0] == IVS:
				if len(info) == 4: # DC Source
					Components[IVS].append(Voltage_source(info[0], info[1], info[2], float(info[3])))
				elif len(info) == 6: # AC Source
					if circuitFreq == 1e-100:
						sys.exit("Frequency of AC Source not specified!!")
					Components[IVS].append(Voltage_source(info[0], info[1], info[2], float(info[4])/2, info[5]))
				# Current Source
			elif info[0][0] == ICS:
				if len(info) == 5: # DC Source
					Components[ICS].append(Current_source(info[0], info[1], info[2], float(info[4])))
				elif len(info) == 6: # AC Source
					if circuitFreq == 1e-100:
						sys.exit("Frequency of AC Source not specified!!")
					Components[ICS].append(Current_source(info[0], info[1], info[2], float(info[4])/2, info[5]))
				# VCVS
			elif info[0][0] == VCVS:
				Components[VCVS].append(vcvs(info[0], info[1], info[2], info[3], info[4], info[5]))
				# VCCS
			elif info[0][0] == VCCS:
				Components[VCCS].append(vcvs(info[0], info[1], info[2], info[3], info[4], info[5]))
				# CCVS
			elif info[0][0] == CCVS:
				Components[CCVS].append(ccvs(info[0], info[1], info[2], info[3], info[4]))
				# CCCS
			elif info[0][0] == CCCS:
				Components[CCCS].append(cccs(info[0], info[1], info[2], info[3], info[4]))
		
		try:
			nodes.remove('GND')
			nodes = ['GND']  + nodes
		except:
			sys.exit("No ground node specified in the circuit!!")
		node_num = {nodes[i]:i for i in range(len(nodes))}
		num_nodes=len(nodes)
		num_VS = len(Components[IVS])+len(Components[VCVS])+len(Components[CCVS])
		print(nodes)
		M = np.zeros((num_nodes+num_VS,num_nodes+num_VS), np.complex128)
		B = np.zeros((num_nodes+num_VS,),np.complex128)

		w = 2*PI*circuitFreq
		if circuitFreq == 1e-100:
			print("\nThis is a DC circuit, \n Frequency = 0.0")
		else:
			print("\nThis is an AC circuit, \n Frequency = {}".format(circuitFreq))
		M[0][0] = 1.0

		for r in Components[RES]:
			if r.node1 != 'GND':
				#print(r.node1)
				#print(node_num[r.node2])
				#print(M[node_num[r.node1]][node_num[r.node1]])
				r.val = float(r.val)
				M[node_num[r.node1]][node_num[r.node1]] += 1/r.val
				M[node_num[r.node1]][node_num[r.node2]] -= 1/r.val
			if r.node2 != 'GND':
				M[node_num[r.node2]][node_num[r.node1]] -= 1/r.val
				M[node_num[r.node2]][node_num[r.node2]] += 1/r.val
		# Capacitor Equations
		for c in Components[CAP]:
			if c.node1 != 'GND':
				M[node_num[c.node1]][node_num[c.node1]] += complex(0, w*c.val)
				M[node_num[c.node1]][node_num[c.node2]] -= complex(0, w*c.val)
			if c.node2 != 'GND':
				M[node_num[c.node2]][node_num[c.node1]] -= complex(0, w*c.val)
				M[node_num[c.node2]][node_num[c.node2]] += complex(0, w*c.val)
		# Inductor Equations
		for l in Components[IND]:
			if l.node1 != 'GND':
				M[node_num[l.node1]][node_num[l.node1]] += complex(0, -1.0/(w*l.val))
				M[node_num[l.node1]][node_num[l.node2]] -= complex(0, -1.0/(w*l.val))
			if l.node2 != 'GND':
				M[node_num[l.node2]][node_num[l.node1]] -= complex(0, -1.0/(w*l.val))
				M[node_num[l.node2]][node_num[l.node2]] += complex(0, -1.0/(w*l.val))
		# Voltage Source Equations	
		for i in range(len(Components[IVS])):
		# Equation accounting for current through the source
			if Components[IVS][i].node1 != 'GND':
				M[node_num[Components[IVS][i].node1]][num_nodes+i] = -1.0
			if Components[IVS][i].node2 != 'GND':
				M[node_num[Components[IVS][i].node2]][num_nodes+i] = 1.0
		# Auxiliary Equations
			M[num_nodes+i][node_num[Components[IVS][i].node1]] = 1.0
			M[num_nodes+i][node_num[Components[IVS][i].node2]] = -1.0
			B[num_nodes+i] = cmath.rect(Components[IVS][i].val, Components[IVS][i].phase*PI/180)

		# Current Source Equations
		for i in Components[ICS]:
			if i.node1 != 'GND':
				B[node_num[i.node1]] = -1*i.val
			if i.node2 != 'GND':
				B[node_num[i.node2]] = i.val
		# VCVS Equations
		for i in range(len(Components[VCVS])):

		# Equation accounting for current through the source
			if Components[VCVS][i].node1 != 'GND':
				M[node_num[Components[VCVS][i].node1]][num_nodes+len(Components[IVS])+i] = -1.0
			if Components[VCVS][i].node2 != 'GND':
				M[node_num[Components[VCVS][i].node2]][num_nodes+len(Components[IVS])+i] = 1.0

			M[num_nodes+len(Components[IVS])+i][node_num[Components[VCVS][i].node1]] = 1.0
			M[num_nodes+len(Components[IVS])+i][node_num[Components[VCVS][i].node2]] = -1.0
			M[num_nodes+len(Components[IVS])+i][node_num[Components[VCVS][i].node3]] = -1.0*Components[VCVS][i].val
			M[num_nodes+len(Components[IVS])+i][node_num[Components[VCVS][i].node4]] = 1.0*Components[VCVS][i].val
		# CCVS Equations
		for i in range(len(Components[CCVS])):
		# Equation accounting for current through the source
			if Components[VCVS][i].node1 != 'GND':
				M[node_num[Components[CCVS][i].node1]][num_nodes+len(Components[IVS])+len(Components[VCVS])+i] = -1.0
			if Components[VCVS][i].node2 != 'GND':
				M[node_num[Components[VCVS][i].node2]][num_nodes+len(Components[IVS])+len(Components[VCVS])+i] = 1.0

			M[num_nodes+len(Components[IVS])+len(Components[VCVS])+i][node_num[Components[CCVS][i].node1]] = 1.0
			M[num_nodes+len(Components[IVS])+len(Components[VCVS])+i][node_num[Components[CCVS][i].node2]] = -1.0
			M[num_nodes+len(Components[IVS])+len(Components[VCVS])+i][num_nodes+len(Components[IVS])+len(Components[VCVS])+i] = -1.0*Components[CCVS][i].val
		
		# VCCS Equations
		for vccs in Components[VCCS]:
			if vccs.node1 != 'GND':
				M[node_num[vccs.node1]][node_num[vccs.node4]]+=vccs.val
				M[node_num[vccs.node1]][node_num[vccs.node3]]-=vccs.val
			if vccs.node2 != 'GND':
				M[node_num[vccs.node2]][node_num[vccs.node4]]-=vccs.val
				M[node_num[vccs.node2]][node_num[vccs.node3]]+=vccs.val
		# CCCS Equations
		for cccs in Components[CCCS]:
			def getIndexIVS(vName):
				for i in range(len(Components[IVS])):
					if Components[IVS][i].name == vName:
						return i
				if cccs.node1 != 'GND':
					M[node_num[cccs.node1]][numNodes+getIndexIVS(cccs.vSource)]-=cccs.val
				if cccs.node2 != 'GND':
					M[node_num[cccs.node2]][numNodes+getIndexIVS(cccs.vSource)]+=cccs.val

		
		try:
			x = np.linalg.solve(M,B)
			#printing node voltages
			for i in range(0,len(nodes)):
				print ("\nVoltage in Node {}: {}".format(nodes[i],x[i]))
		

			# Formatting Output Data
			for i in range(len(nodes)  , len(nodes)+len(Components[IVS])):
				for v in Components[IVS]:
					print("\nCurrent through {} is: {}".format(v.name,x[i]))
			for i in range(len(nodes)+len(Components[IVS])  ,  len(nodes)+len(Components[IVS])+len(Components[VCVS])):
				for v in Components[VCVS]:
					print("\nCurrent through {} is: {}".format(v.name,x[i]))
			for i in range(len(nodes)  + len(Components[IVS]) + len(Components[VCVS]) ,  len(nodes) + len(Components[IVS]) + len(Components[VCVS]) + len(Components[CCVS])):
				for v in Components[CCVS]:
					print("\nCurrent through {} is: {}".format(v.name,x[i]))
		except:
			sys.exit("Singular Matrix formed, Please check values entered")
		
except IOError:
	sys.exit("Invalid file")
