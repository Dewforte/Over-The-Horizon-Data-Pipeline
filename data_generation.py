import numpy as np
import pandas as pd
from scipy.special import erfc

# Reproducibility
np.random.seed(42)

# Constants
c = 3e8  # Speed of light (m/s)
k_B = 1.38e-23  # Boltzmann constant (J/K)
T_sys = 290  # System temperature (K)
noise_floor_dbm = -100  # Noise floor (dBm)
eta = 0.1  # Ionization efficiency (fraction)
N_A = 6.022e23  # Avogadro's number

# Number of samples
n_samples = 5000

# Sample base features
altitude_km = np.random.uniform(100, 150, n_samples)  # km
release_mass = np.random.uniform(0, 0.05, n_samples)  # kg
cloud_age = np.random.uniform(1, 5400, n_samples)  # seconds
diffusion_coefficient = 1 + (altitude_km - 100) * 0.05  # m^2/s
tx_frequency = np.random.uniform(30e6, 300e6, n_samples)  # Hz
tx_power = np.random.uniform(10, 100, n_samples)  # dBm
tx_gain = np.random.uniform(0, 15, n_samples)  # dBi
rx_gain = np.random.uniform(0, 15, n_samples)  # dBi
distance = np.random.uniform(50e3, 1000e3, n_samples)  # meters
wind_speed = np.random.uniform(0, 50, n_samples)  # m/s
solar_angle = np.random.uniform(0, 90, n_samples)  # degrees
kp_index = np.random.uniform(0, 9, n_samples)  # Kp index

# Derived features
D = diffusion_coefficient
t = cloud_age
Q = eta * release_mass * N_A * 1  # electrons (1 electron per atom)

cloud_radius = np.sqrt(4 * D * t)  # meters
n_e = Q / ((4 * np.pi * D * t) ** 1.5)  # electrons/m^3
f_c = 9e3 * np.sqrt(n_e)  # Hz

# Wavelength and path loss
wavelength = c / tx_frequency
path_loss_db = 20 * np.log10(4 * np.pi * distance / wavelength)
received_power_db = tx_power + tx_gain + rx_gain - path_loss_db

# Convert from dBm to linear scale (milliwatts)
received_power_mw = 10 ** (received_power_db / 10)

# Apply Rayleigh fading (scale=1.0 ensures unit mean power)
rayleigh_gain = np.random.rayleigh(scale=1.0)

# Apply fading in linear scale
faded_power_mw = received_power_mw * rayleigh_gain

# Convert back to dBm
faded_power_db = 10 * np.log10(faded_power_mw)

# Update received power to include fading
received_power_db = faded_power_db


# SNR and BER (BER is for reference only)
snr_db = received_power_db - noise_floor_dbm
snr_linear = 10 ** (snr_db / 10)
ber = 0.5 * erfc(np.sqrt(snr_linear))  # BPSK in AWGN

# Delay spread and Doppler shift
delay_spread_us = cloud_radius / c * 1e6  # in microseconds
doppler_shift_Hz = (wind_speed / c) * tx_frequency

# Create DataFrame
df = pd.DataFrame({
    "altitude_km": np.floor(altitude_km),
    "release_mass": release_mass,
    "cloud_age": cloud_age,
    "diffusion_coefficient": diffusion_coefficient,
    "tx_frequency": tx_frequency,
    "tx_power": tx_power,
    "tx_gain": tx_gain,
    "rx_gain": rx_gain,
    "distance": distance,
    "wind_speed": wind_speed,
    "solar_angle": solar_angle,
    "kp_index": kp_index,
    "cloud_radius": cloud_radius,
    "n_e": n_e,
    "f_c": f_c,
    "received_power": received_power_db,
    "snr_db": snr_db,
    "delay_spread_us": delay_spread_us,
    "doppler_shift_Hz": doppler_shift_Hz,
    "ber": ber  # Not to be used as ML target
})

# Export
df.to_csv("updated_plasma_oth_dataset.csv", index=False)
print("✅ Dataset saved as 'updated_plasma_oth_dataset.csv'")
