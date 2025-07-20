#parameters for neuron and ultrasound simulation model

model_parameters = {

        #Neuron parameters

"species" : "Squid" ,

# morphology
"soma_length" : 12.6157, #length of soma (meters)

"soma_diameter" : 12.6157, #diameter of soma (meters)

"dend_length" : 200, #length of dendrite (meters)
 
"dend_diameter" : 1, #diameter of dendrite (meters)

# biophysics

"axial_res" : 100, #axial resistance (Ohm*cm)

"membrane_cap" : 1, #membrane capactiance in (micro farads/cm^s) 

"Na_conductance" :  0.12, #sodium conductance (siemens)

"K_conductance" :  0.036, #potassium conductance (siemens)

"leak_conductance": 0.0003, #leak conductance (siemens)

"rev_potential" : -54.3, #reversal potential in (mV)

"passive_conductance" : 0.001, #passive conductance (siemens)

"leak_rev_potential": -65,  #leak reversal potenatial (mV)

# synapse_tau : 2 * ms 

#synapse and stimulus
"number_neurons" : 5, #number of cells

"stimulus_weight" : 0.04, #weight of the stimulus 

"stimulus_time" : 9, #time of the sitmulus(in ms)

"stimulus_delay" : 1, #delay of the stimulus (in ms)

"synapse_weight" : 0.01, #synaptic weight

"synapse_delay" : 5, #delay of the synapse

"radius" : 50, #radius of the network



        #ultrasound parameters
#arc properties

"arc_radius" : 100e-8, #radius of arc (meters)

"arc_diameter" : 8e-8, #diameters of arc (meters)

"ring_radius" : 50e-7, #radius of ring (meters)

"num_ultrasound_detectors" : 3, #number of detectors

#grid properties
"grid_x" : 256, #width of the grid

"grid_y" : 256, #height of the grid

"displacement_x" : 0.5e-7, #width of the displacement 

"displacement_y" : 0.5e-7, #height of the displacement

#medium properties
"sound_speed" : 1540, #sound speed in (meters/second)


 }