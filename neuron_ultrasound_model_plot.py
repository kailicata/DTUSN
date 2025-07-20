# Import plotting and numerical libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Plot a single neuron in 3D
def neuron_plot(cell):
    xn, yn, zn = np.array([]), np.array([]), np.array([])
    #these are the confocal microscopy coordinates of the neuron if the ring of 5 neruons 
    # Collect all 3D coordinates from each section of the neuron
    for sec in cell.external_all:
        for p in sec:
            xn = np.append(xn, p[0])
            yn = np.append(yn, p[1])
            zn = np.append(zn, 0)

    # 3D scatter plot of neuron structure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xn, yn, zn, marker='o', color=(0, 1, 0))
    ax.set_xlabel('neuron cell len (micrometers)')
    ax.set_ylabel('neuron cell len (micrometers)')
    ax.set_zlabel('neuron cell len (micrometers)')
    plt.title("Single Neuron Morphology")
    plt.show()

# Plot a ring of neurons in 3D
def ring_of_neurons(ring):
    xn, yn, zn = np.array([]), np.array([]), np.array([])

    # Collect 3D coordinates for all neurons in the ring
    #these are the confocal microscopy coordinates of the neuron of the single neruon 
    for cell in ring.cells:
        for sec in cell.external_all:
            for p in sec:
                xn = np.append(xn, p[0])
                yn = np.append(yn, p[1])
                zn = np.append(zn, 0)

    # 3D scatter plot of all neurons
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xn, yn, zn, marker='o', color=(0,1,0))
    ax.set_xlabel('neuron cell len (micrometers)')
    ax.set_ylabel('neuron cell len (micrometers)')
    ax.set_zlabel('neuron cell len (micrometers)')
    plt.show()

# Combined plot of neuron ring, ultrasound source, sensors, and pressure
def neuron_and_ultrasound_plot(ring, source_points, sensor_data, sensor_location, p_first_dendrite):
    # Get neuron coordinates
    xn, yn, zn = np.array([]), np.array([]), np.array([])
    for cell in ring.cells:
        for sec in cell.external_all:
            for p in sec:
                xn = np.append(xn, p[0])
                yn = np.append(yn, p[1])
                zn = np.append(zn, 0)

    # Get ultrasound source points (nonzero values in 2D grid)
    xs, ys = np.where(source_points)
    zs = np.zeros(len(xs))  # z is zero since it's a 2D surface

    # Get sensor locations
    yd, xd = np.where(sensor_location)
    zd = np.zeros(len(xd))  # z is zero here too

    # Initialize 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Get first pressure point info for debugging
    pressure_intensity = sensor_data[0][0]
    pressure_point_x = np.where(sensor_data)[1]
    pressure_point_y = np.where(sensor_data)[0]
    print("The x-coordinate of the initial pressure point is " + str(pressure_point_x))
    print("The y-coordinate of the initial pressure point is " + str(pressure_point_y))
    print("pressure intensity" + str(pressure_intensity))  # Pa

    # Plot pressure points from sensor data
    ax.scatter(pressure_point_x, pressure_point_y, color=(0, 0, 1))
    ax.set_xlabel('(sensor data) The pressure points grid - rows (1207)')
    ax.set_ylabel('(sensor data) The pressure points grid - coloumns (140)')
    plt.show()

    # Plot everything in 3D
    pdx, pdy, pdz = np.array(p_first_dendrite[0]), np.array(p_first_dendrite[1]), np.array(p_first_dendrite[2])
    ax.scatter(xn, yn, zn, marker='o', color=(0,1,0))     # Neuron = green
    ax.scatter(xs, ys, zs, marker='o', color=(1,0,0))     # Ultrasound source = red
    ax.scatter(xd, yd, zd, marker='o', color=(0,0,1))     # Sensor = blue
    ax.scatter(pdx, pdy, pdz, marker='o', color=(0,0,0))  # Pressure = black

    ax.set_xlabel('neuron cell len (micrometers)')
    ax.set_ylabel('neuron cell len (micrometers)')
    ax.set_zlabel('neuron cell len (micrometers)')
    plt.show()

# Plot overlay of different simulation masks (sensor, pressure, PML)
def plot_simulation_masks(sensor, logical_p0, pm1_mask):
    sensor_val = sensor.mask * 1
    logical_p0_val = logical_p0 * 2
    pm1_mask_val = pm1_mask * 3

    # Combine into a single matrix for plotting
    combined_mask = sensor_val + logical_p0_val + pm1_mask_val
    combined_mask = np.flipud(combined_mask)  # Flip for display

    # Define custom colormap for components
    colors = [(1,1,1), (233/255,131/255,0), (253/255,221/255,92/255), (0.8, 0.8, 0.8)]
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    labels = {
        0: "None",
        1: "Sensor",
        2: "Initial pressure p0",
        3: "PML mask",
    }

    bounds = np.linspace(0, len(labels), len(labels) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Create figure and plot
    fig, ax = plt.subplots(layout="tight", figsize=(10, 4))
    ax.pcolormesh(combined_mask, cmap=cmap, shading="auto")
    ax.set_title("Simulation of the Neural Ultrasonic Interaction (DTUSN)")
    ax.set_ylabel("Simulation Components [-]")
    plt.axis("image")

    # Add legend colorbar
    ax2 = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing="proportional", ticks=bounds, boundaries=bounds)
    ax2.set_yticks([(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds) - 1)])
    ax2.set_yticklabels(list(labels.values()))
    plt.show()

# Show ultrasound detector data as an image
def plot_sensor_data_image(combined_sensor_data):
    fig, ax = plt.subplots()
    im = ax.imshow(combined_sensor_data, aspect="auto", cmap="viridis", interpolation="none")
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel("Detector Number")
    ax.set_title("Arc detector")
    fig.colorbar(im, ax=ax)
    plt.show()

# Plot pressure trace over time for the first detector
def plot_sensor_trace(kgrid, combined_sensor_data):
    plt.figure()
    plt.plot(kgrid.t_array.squeeze() * 1e6, combined_sensor_data[0, :], label="Arc detectors")
    plt.xlabel(r"Time [$\mu$s]")
    plt.ylabel("Pressure [Pa]")
    plt.legend()
    plt.show()

# Plot voltage of soma over time for first cell
def soma_voltage_over_time(ring, mp, h, mV, ms):
    shape_window = h.PlotShape(True)
    shape_window.show(0)

    # Record and simulate
    t = h.Vector().record(h._ref_t)
    h.finitialize(-65 * mV)
    h.continuerun(100 * ms)

    plt.plot(t, ring.cells[0].soma_v)
    plt.title(f"{mp['species']} Soma Voltage Over Time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.show()

# Raster plot of spike times for each cell
def plot_spike_times(ring, mp):
    plt.figure()
    for i, cell in enumerate(ring.cells):
        plt.vlines(list(cell.spike_times), i + 0.5, i + 1.5)
    plt.title(f"{mp['species']} Spike Times of Ring Network Cells")
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell Index")
    plt.show()

# Compare spike times under different synaptic weights
def plot_spike_times_with_synaptic_weights(ring, mp, h, mV, ms, RingClass, cell_data):
    ring = RingClass(mp, N=6, syn_w=syn_w, cell_data=cell_data)
    plt.figure()
    for syn_w, color in [(0.01, "black"), (0.005, "red")]:
        ring = Ring(mp, N=6, syn_w=syn_w)
        h.finitialize(-65 * mV)
        h.continuerun(100 * ms)
        for i, cell in enumerate(ring.cells):
            plt.vlines(list(cell.spike_times), i + 0.5, i + 1.5, color=color)
    plt.title(f"{mp['species']} Spike Times of Ring Network Cells with Different Synaptic Weights")
    plt.xlabel("Time (ms)")
    plt.ylabel("Cell Index")
    plt.show()