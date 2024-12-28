
# to import the neuron module
import neuron 

#both h and rxd are submodules from neuron
from neuron import h,rxd

#the units of meausrment are micro seconds and micro volts (micro = 10^-6)
from neuron.units import ms, mV

#defining the soma -> section is used to represent the specific sectio  of the soma model
soma = h.Section(name="soma")

#topolgy will display the structure or body of the soma 
h.topology()

#psection means properties of the section and will list all the properties of whats stored in the soma
soma.psection()

#this would give us the specific property of the soma, in this case the length 
soma.psection()['morphology']['L']

#this is an alternative way to find the length
soma.L

#we can set the soma to a different length and diameter 
soma.L = 20
soma.diam = 20

#the dir funtion lists the python methoids and varibales associated with whats inside the paranthesis, in this case the soma
dir(soma)

#the textwrap will make all the listed properties more readable and not go offscreen 
import textwrap
#this will run the dir function but with textwrap so it looks cleaner 
print(textwrap.fill(','.join(dir(h))))

#will display the comments of the program
help(soma.connect)

#this will insert the Hodgkin-Huxley channels into the somas membrane 
soma.insert('hh')

#this will tell is what a varibale is, in this case its a section 
print("type(soma) = {}".format(type(soma)))
#this will tell is what a varibale is, in this case its a segment 
print("type(soma(0.5)) = {}".format(type(soma(0.5))))

#this is setting the 0.5 segment of the soma with a hodglen huxley channel to the varibale mech
mech = soma(0.5).hh
#this will print the properties of the segment 
print(dir(mech))

#this will print the specific property of the soma segment, in this case the gkbar 
print(mech.gkbar)
#this is an alternative way to print it
print(soma(0.5).hh.gkbar)

#iclamp is a tool used to chnage the current and therefore affect the membrane 
#this will inset an iclmap in the soma segment 
iclamp = h.IClamp(soma(0.5))

#this will give us the properties of the iclamp without the python methods
print([item for item in dir(iclamp) if not item.startswith('_')])

#this will set the properties of the icmalp to different values 
iclamp.delay = 2
iclamp.dur = 0.1
iclamp.amp = 0.8

#this is lisitng all the properties of the section of the soma 
soma.psection()

#this will record the membrane potential of the soma segemnt and store them into v
v = h.Vector().record(soma(0.5).ref_v)
#this will record the time of the soma segment and stor them into t 
t = h.Vector().record(h.ref_t)

#this will advance the time by one increment 
h.load_file('stdrun.hoc')

#this will set the simukation to have a resting membrane potential  of -65 mV
h.finitialize(-65 *mV)

#this will make it run for 40 micro seconds 
h.continuerun(40* ms)

#json is the most commonly used way to store data and plot
import json

#this will list and store all the data in an organized way, time will be showen in the t list and the potential will be shown in the v list
with open('data.json','w')as f:
    json.dump({'t':list(t), 'v':list(v)},f, indent = 4)

#when reading it does not have to be expleicitly stated with r. this will let us read the file/data
with open('data.json') as f:
    data = json.load(f)
tnew = data['t']
vnew = data['v']

#this will plot our data
plt.figure()
plt.plot(tnew,vnew)
plt.xlabel('t (ms)')
plt.ylabel('v (mV)')
plt.show()