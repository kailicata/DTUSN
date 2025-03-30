import os
import random
import numpy as np
from pathlib import Path
from neuron_ultrasound_model0 import classify_action_potential
from model_parameters import model_parameters as mp 




num_pressure_points = 1207



def generate_sample(num_pressure_points=num_pressure_points):
    # Base values
    p = mp.copy()
    p["soma_length"] = 12.6157 * random.uniform(0.9, 1.1)
    p["soma_diameter"] = 12.6157 * random.uniform(0.9, 1.1)
    p["dend_length"] = 200 * random.uniform(0.95, 1.05)
    p["dend_diameter"] = 1 * random.uniform(0.9, 1.1)
    
    p["sodium_conductance"] = 0.12 * random.uniform(0.95, 1.1)
    p["potassium_conductance"] = 0.036 * random.uniform(0.95, 1.1)
    # Biophysics parameters
    #axial_resistance = 100 * random.uniform(0.95, 1.05)
    p["membrane_capacitance"] = 1 * random.uniform(0.95, 1.05)
    
    # Conductance variations based on class
    is_action_potential = classify_action_potential(p)
    if is_action_potential:
        p["axial_resistance"] = 100 * random.uniform(1.5, 1.7)

    else:
        p["axial_resistance"] = 100 * random.uniform(0.3,0.5)
    
    p["leak_conductance"] = 0.003 * random.uniform(0.95, 1.05)
    p["reversal_potential"] = -54.3 * random.uniform(0.98, 1.02)
    p["passive_conductance"] = 0.001 * random.uniform(0.95, 1.05)
    p["leak_reversal_potential"] = -65 * random.uniform(0.98, 1.02)
    
    # Generate pressure points with variations
    base_points = np.linspace(0, 4*np.pi, num_pressure_points)
    
    # Define ranges for each class to ensure separation
    if is_action_potential:
        # Action potential: Range approximately 1500-3500
        base = 2500
        amplitude = 1000
        noise_std = 150
    else:
        # No action potential: Range approximately 500-1500
        base = 1000
        amplitude = 500
        noise_std = 100
    
    # Generate the waveform
    frequency = 1.2 if is_action_potential else 0.8
    pressure_points = base + amplitude * np.sin(frequency * base_points) + np.random.normal(0, noise_std, num_pressure_points)
    
    # Create the sample text
    sample = f"""This is the neurons morphology: 
Soma length: {p["soma_length"]} meters
Soma diameter: {p["soma_diameter"]} meters
Dendrite length: {p["dend_length"]} meters
Dendrite diameter: {p["dend_diameter"]} meter
This is the neurons biophysics: 
Axial resistance: {p["axial_resistance"]} Ohm*cm 
Membrane capacitance: {p["membrane_capacitance"]} microfarads/cm^2
Sodium Conductance: {p["sodium_conductance"]} siemens
Potassium Conductance: {p["potassium_conductance"]} siemens 
Leak conductance: {p["leak_conductance"]} siemens 
Reversal potential: {p["reversal_potential"]} millivolts 
Passive conductance: {p["passive_conductance"]} siemens
Leak reversal potential: {p["leak_reversal_potential"]} millivolts 
Ultrasound stimulation: 
Pressure points:{pressure_points.tolist()}"""
    
    return sample

def generate_dataset(num_samples_per_class, num_pressure_points=num_pressure_points):
    # Create directories if they don't exist
    base_dir = Path(os.getcwd())
    data_set_dir = base_dir / 'data_set'
    ap_dir = data_set_dir / 'action_potential'
    no_ap_dir = data_set_dir / 'no_action_potential'
    
    # Create the directory structure
    data_set_dir.mkdir(exist_ok=True)
    ap_dir.mkdir(exist_ok=True)
    no_ap_dir.mkdir(exist_ok=True)
    
    # Generate samples for each class
    for i in range(num_samples_per_class):
        # Action potential samples
        ap_sample = generate_sample(num_pressure_points=num_pressure_points)
        ap_file = ap_dir / f'sample_{i+1}.txt'
        with open(ap_file, 'w') as f:
            f.write(ap_sample)
        
        # No action potential samples
        no_ap_sample = generate_sample(num_pressure_points=num_pressure_points)
        no_ap_file = no_ap_dir / f'sample_{i+1}.txt'
        with open(no_ap_file, 'w') as f:
            f.write(no_ap_sample)

def save_dataset(data_dir='data_set', save_path='neuron_dataset.npz'):
    """Save the generated dataset in IMDB-compatible format."""
    base_dir = Path(os.getcwd()) / data_dir
    ap_dir = base_dir / 'action_potential'
    no_ap_dir = base_dir / 'no_action_potential'
    
    # Lists to store data
    x_data = []
    y_data = []
    
    def extract_pressure_points(content):
        # Find the pressure points line and extract the values
        start = content.find('Pressure points:[') + len('Pressure points:[') 
        end = content.find(']', start)
        points_str = content[start:end]
        # Convert string of numbers to list of floats
        points = [float(x.strip()) for x in points_str.split(',')]
        # Convert to IMDB-style word indices (3 to 9999)
        points = np.array(points)
        points = (points - points.min()) / (points.max() - points.min()) * 9996 + 3
        return points.astype(int)  # Convert to integers for word-like tokens
    
    # Read action potential samples (positive sentiment)
    for file in ap_dir.glob('sample_*.txt'):
        with open(file, 'r') as f:
            content = f.read()
            pressure_points = extract_pressure_points(content)
            x_data.append(pressure_points)
            y_data.append(1)  # Positive sentiment
    
    # Read no action potential samples (negative sentiment)
    for file in no_ap_dir.glob('sample_*.txt'):
        with open(file, 'r') as f:
            content = f.read()
            pressure_points = extract_pressure_points(content)
            x_data.append(pressure_points)
            y_data.append(0)  # Negative sentiment
    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    # Randomly shuffle the data
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = x_data[indices]
    y_data = y_data[indices]
    
    # Split into train and validation sets (80-20 split)
    split_idx = int(len(x_data) * 0.8)
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_val = x_data[split_idx:]
    y_val = y_data[split_idx:]
    
    # Save the dataset
    np.savez(save_path,
             x_train=x_train, y_train=y_train,
             x_val=x_val, y_val=y_val)
    
    print(f"Dataset shape: x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"Sample sequence: {x_train[0][:10]}...")
    
    return (x_train, y_train), (x_val, y_val)

def load_data(path='neuron_dataset.npz', num_words=None):
    """Load the saved dataset in IMDB format.
    
    Args:
        path: Path to the dataset file
        num_words: Maximum number of words (ignored, kept for IMDB compatibility)
    
    Returns:
        Tuple of (x_train, y_train), (x_val, y_val) in IMDB format
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found. Generate the dataset first.")
    
    data = np.load(path)
    return (data['x_train'], data['y_train']), (data['x_val'], data['y_val'])

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate neuron sample datasets')
    parser.add_argument('--num_samples_per_class', type=int, default=10,
                        help='Number of samples to generate per class')
    parser.add_argument('--save', action='store_true',
                        help='Save the generated dataset')
    parser.add_argument('--save_path', type=str, default='neuron_dataset.npz',
                        help='Path to save the dataset')
    
    args = parser.parse_args()
    
    # Generate the dataset
    generate_dataset(args.num_samples_per_class)
    
    # Save the dataset if requested
    if args.save:
        print(f"Saving dataset to {args.save_path}...")
        (x_train, y_train), (x_val, y_val) = save_dataset(save_path=args.save_path)
        print(f"Dataset saved with {len(x_train)} training samples and {len(x_val)} validation samples")