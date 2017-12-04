def add_params_und():
	from awot.io.name_maps_flight import get_name_map
	und_map = get_name_map('citation')
	#add cloud water content from csi probe
	und_map['csi_cwc'] = 'Cloud Water Content from the CSI probe with Correction [g/m^3] {Calibration: TDL slope = 1.5070000 TDL offset = 0.025000000 Out-of-Cloud slope = 1.0000000 Out-of-Cloud offset = 0.0000000'
	#add 2DS hor particle conc > 105 mu m
	und_map['2DShor_NormPC105'] = 'TwoDS (SN: 012) Horizontal Total Normalize Particle Concentration of Particles Greater than 105 microns [#/m^3]'
	#add 2DS vert particle conc > 105 mu m
	und_map['2DSver_NormPC105'] = 'TwoDS (SN: 012) Vertical Total Normalize Particle Concentration of Particles Greater than 105 microns [#/m^3]'
	#add 2DS hor particle conc all
	und_map['2DShor_NormAll'] = 'TwoDS (SN: 012) Horizontal Total Normalize Particle Concentration for All Bin Sizes [#/m^3]'
	#add 2DS hor particle conc all
	und_map['2DSver_NormAll'] = 'TwoDS (SN: 012) Vertical Total Normalize Particle Concentration for All Bin Sizes [#/m^3]'
	#HVPS3 hor All
	und_map['HVPS3_hor'] = 'HVPS3 (SN: 04) Horizontal Total Normalize Particle Concentration for All Bin Sizes [#/m^3]'
	#HVPS3 vert ALl
	und_map['HVPS3_vert'] = 'HVPS3 (SN: 07) Vertical Total Normalize Particle Concentration for All Bin Sizes [#/m^3]'
	#Add iwc for nev
	und_map['Nev_IWC'] = 'Ice Water Content based on the Nevzorov Probe measurement (beta = 0.110000) [g/m^3]'
	    
	return(und_map)
