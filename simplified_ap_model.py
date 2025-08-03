from neuron import h, gui
import numpy as np
from neuron.units import ms, mV


h.load_file("stdrun.hoc")

class Cell:
    def __init__(self, gid, x, y, z, theta):
        self._gid = gid
        self._setup_morphology()
        self.all = self.soma.wholetree()
        self._setup_biophysics()
        self.x = self.y = self.z = 0  # <-- NEW
        h.define_shape()
        self._rotate_z(theta)  # <-- NEW
        self._set_position(x, y, z)  # <-- NEW

    def __repr__(self):
        return "{}[{}]".format(self.name, self._gid)

    # everything below here is NEW

    def _set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(
                    i,
                    x - self.x + sec.x3d(i),
                    y - self.y + sec.y3d(i),
                    z - self.z + sec.z3d(i),
                    sec.diam3d(i),
                )
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))



class BallAndStick(Cell):
    name = "BallAndStick"

    def _setup_morphology(self):
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1

    def _setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert("hh")
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert("pas")
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV



def create_n_BallAndStick(n, r):
    """n = number of cells; r = radius of circle"""
    cells = []
    for i in range(n):
        theta = i * 2 * h.PI / n
        cells.append(BallAndStick(i, h.cos(theta) * r, h.sin(theta) * r, 0, theta))
    return cells



def simulate_simplified_neuron_voltage(voltage_from_frequency, duration):

    my_cells = create_n_BallAndStick(7, 50)

    stim = h.NetStim()  # Make a new stimulator

    ## Attach it to a synapse in the middle of the dendrite
    ## of the first cell in the network. (Named 'syn_' to avoid
    ## being overwritten with the 'syn' var assigned later.)
    syn_ = h.ExpSyn(my_cells[0].dend(0.5))

    stim.number = 1
    stim.start = 9
    ncstim = h.NetCon(stim, syn_)
    ncstim.delay = 1 * ms
    ncstim.weight[0] = 0.04  # NetCon weight is a vector.

    syn_.tau = 2 * ms

    recording_cell = my_cells[0]
    soma_v = h.Vector().record(recording_cell.soma(0.5)._ref_v)
    dend_v = h.Vector().record(recording_cell.dend(0.5)._ref_v)
    t = h.Vector().record(h._ref_t)

    #voltage_from_frequency = 0
    threshold_voltage_adjusted = -65   # Adjust threshold based on frequency
    print("Voltage after Ultrasound: " + str(threshold_voltage_adjusted))
    h.finitialize((threshold_voltage_adjusted)* mV)
    h.continuerun(duration * ms)

    # Convert NEURON vectors to NumPy arrays
    soma_v_np = np.array(soma_v)
    t_np = np.array(t)
    # Find time of peak voltage
    peak_index = np.argmax(soma_v_np)
    peak_voltage = soma_v_np[peak_index] 
    peak_voltage = peak_voltage + voltage_from_frequency
    peak_time = t_np[peak_index]

    print(f"Peak soma voltage: {peak_voltage:.3f} mV at time: {peak_time:.3f} ms")


    import matplotlib.pyplot as plt

    plt.plot(t, soma_v, label="soma(0.5)")
    plt.plot(t, dend_v, label="dend(0.5)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.ylim(-80, 80)
    plt.legend()
    plt.show()
    

    return soma_v, dend_v


if __name__ == "__main__":
    v = 60 
    duration = 100
    simulate_simplified_neuron_voltage(v, duration)

