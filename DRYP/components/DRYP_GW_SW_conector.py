import numpy as np

def call_update_soil(z, zroot_u, zroot_l, 
		theta_sat_u, theta_fc_u, theta_u,
		theta_sat_l, theta_fc_l, theta_l,
		deltaS, head, Sy):
	"""
	"""
	#head_out = np.zeros_like(head)
	#print(type(theta_sat_u))
	if theta_sat_u is not np.ndarray:		
		for i, ihead in enumerate(head): 
			head[i] = update_2layer_soil(z[i], zroot_u[i], zroot_l[i],
			theta_sat_u, theta_fc_u, theta_u,
			theta_sat_l[i], theta_fc_l[i], theta_l[i],
			deltaS[i], ihead, Sy[i])
	else:
		for i in enumerate(head): 
			head[i] = update_2layer_soil(z[i], zroot_u[i], zroot_l[i],
			theta_sat_u[i], theta_fc_u[i], theta_u[i],
			theta_sat_l[i], theta_fc_l[i], theta_l[i],
			deltaS[i], head[i], Sy[i])
	#print(head)
	return head
	
def update_2layer_soil(z, zroot_u, zroot_l,
	theta_sat_u, theta_fc_u, theta_u,
	theta_sat_l, theta_fc_l, theta_l,
	deltaS, head, Sy):
	""" Function to calculate soil gorundwater interaction,
	this function update soil storage when the water table
	raise or decrease.
	INPUTS:
	-------
	z:	surface elevation
	zroot_u:	root elevation top layer
	zroot_l:	root elevation bottom layer
	theta_sat_u:	
	theta_fc_u:
	theta_u:
	theta_sat_l:
	theta_fc_l:
	theta_l:
	delta:	change in storage
	head:	water table elevation
	Sy:		aquifer specific yield
	OUTPUTS:
	--------
	h: water table elevation
	"""
	
	if deltaS < 0:
		
		# when water table decrease
		if head > zroot_u:
			deltaSu = (head-zroot_u)*(theta_sat_u-theta_fc_u)
			deltaSl = (zroot_u-zroot_l)*(theta_sat_l-theta_fc_l)
		else:
			deltaSu = 0
			deltaSl = (head-zroot_l)*(theta_sat_l-theta_fc_l)
			if head < zroot_l:
				deltaSl = 0
				
		deltaS = np.abs(deltaS)		
		deltaSp = deltaS - deltaSu
		#print(deltaSp, deltaSl, deltaSl)
		# update head elevation
		if deltaSu > 0:
			# initial water table located in the upper soil layer
			if deltaSp < 0:
				# water table always within the upper soil layer
				head = head - deltaS/(theta_sat_u-theta_fc_u)
			else:
				deltaSpp = deltaS - deltaSu - deltaSl
				if deltaSpp < 0:
					# water table falls to lower soil layer
					head = zroot_u - (deltaS-deltaSu)/(theta_sat_u-theta_fc_u)
				else:
					# water table fails below the lower soil layer
					head = zroot_l - deltaSpp/(Sy)
		else:
			# water table located below the upper soil layer
			deltaSp = deltaS - deltaSl
			if deltaSl > 0:
				# water table always located below the upper soil layer
				if deltaSp < 0:
					# water table always within the lower soil layer
					head = head - (deltaS)/(theta_sat_l-theta_fc_l)
				else:
					#water table falls below the lower layer
					head = zroot_u - (deltaS-deltaSl)/Sy
			else:
				# water table allways in the aquifer
				head = head - deltaS/Sy
		#print(head)	
	else:
		# when water table increases
		if head > zroot_u:
			deltaSl = 0
			deltaSa = 0
		else:
			if head > zroot_l:
				deltaSa = 0
				deltaSl = (zroot_u-head)*(theta_sat_l-theta_l)
			else:
				deltaSl = (zroot_u-zroot_l)*(theta_sat_l-theta_l)
				deltaSa = (zroot_l-head)*Sy
		
		# update water table
		deltaSp = deltaS - deltaSa
		
		if deltaSa > 0:
			# initial water table in the aquifer
			if deltaSp < 0:
				# water table always below the lower soil layer
				head = head + deltaS/Sy
			else:
				# water table always above the aquifer
				deltaSpp = deltaSp - deltaSl
				if deltaSpp > 0:
					# water table rises above the lower layer
					head = zroot_u + (deltaS-deltaSa)/(theta_sat_l-theta_l)
				else:
					# water table rises above the aquifer
					head = zroot_l + deltaSpp/(theta_sat_u-theta_u)
		else:
			# initial water table above the aquifer
			if deltaSl > 0:
				deltaSpp = deltaS - deltaSl
				if deltaSpp > 0:
					# water table within the lower layer
					head = head + deltaS/(theta_sat_l-theta_l)
				else:
					# water table rises to upper layer
					head = zroot_u + deltaSpp/(theta_sat_u-theta_u)
			else:
				# water table always in the upper layer
				head = head + deltaS/(theta_sat_u-theta_u)
				
	return head