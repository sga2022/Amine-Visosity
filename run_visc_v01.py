import numpy as np

# Define constants and parameters
kb = 1.38064852e-23  # Boltzmann constant (J/K)
T = 298  # Temperature (K)
rho = 1000  # Density of the liquid (kg/m^3)
L = 3.39e-10  # Length of the simulation box (m)
rcut = 1.2e-9  # Cutoff radius for the LJ potential (m)
dt = 1e-15  # Time step (s)
steps = 5000  # Number of simulation steps
# steps = 10000  # Number of simulation steps
######### 
freq = 10  # Frequency of data collection (every freq steps)
nparticles = 2000  # Number of particles in the simulation box

# Define LJ parameters and charges for each species
# Monoethanolamine
m1 = 61.08e-3 / 6.022e23  # Mass (kg)
sigma1 = 3.96e-10  # Diameter (m)
epsilon1 = 0.50e-20  # Energy (J)
q1 = 0  # Charge (C)
# Carbamate ion of monoethanolamine
m2 = 61.08e-3 / 6.022e23  # Mass (kg)
sigma2 = 3.96e-10  # Diameter (m)
epsilon2 = 0.35e-20  # Energy (J)
q2 = -1.0 * 1.60e-19  # Charge (C)

# Initialize positions, velocities and forces randomly
positions = L * np.random.rand(nparticles, 3)
velocities = np.random.normal(0, np.sqrt(kb*T/m1), (nparticles, 3))
forces = np.zeros((nparticles, 3))

# Define the LJ potential function and its derivative
def lj(r, sigma, epsilon):
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def lj_derivative(r, sigma, epsilon):
    return 24 * epsilon * (2*(sigma/r)**12 - (sigma/r)**6) / r

# Define the Coulomb potential and its derivative
def coulomb(r, q1, q2):
    return 1 / (4 * np.pi * epsilon0) * q1 * q2 / r

def coulomb_derivative(r, q1, q2):
    return -1 / (4 * np.pi * epsilon0) * q1 * q2 / r**2

# Compute the potential energy and its derivative for all pairs of particles
def compute_potential_energy_and_forces(positions, forces):
    potential_energy = 0
    for i in range(nparticles):
        for j in range(i+1, nparticles):
            r = np.linalg.norm(positions[i,:] - positions[j,:])
            if r < rcut:
                # Compute LJ potential and forces
                if i < j:
                    if (i < 1000 and j < 1000) or (i >= 1000 and j >= 1000):
                        sigma = sigma1
                        epsilon = epsilon1
                    else:
                        sigma = (sigma1 + sigma2) / 2
                        epsilon = np.sqrt(epsilon1 * epsilon2)
                    potential_energy += lj(r, sigma, epsilon)
                    force = lj_derivative(r, sigma, epsilon) * (positions[i,:] - positions[j,:]) / r
                    forces[i,:] += force
                    forces[j,:] -= force
    return potential

# Run the simulation
viscosity = 0
for i in range(steps):
    # Update positions using velocity Verlet algorithm
    positions += velocities * dt + 0.5 * forces / m1 * dt**2
    # Apply periodic boundary conditions
    positions = np.mod(positions + L/2, L) - L/2
    # Compute forces and potential energy
    potential_energy = compute_potential_energy_and_forces(positions, forces)
    # Update velocities
    velocities += 0.5 * forces / m1 * dt
    # Compute the Green-Kubo integral
    if i % freq == 0 and i > 0:
        gk_integrand = np.dot(velocities.flatten(), velocities.flatten())
        viscosity += gk_integrand * dt / (3 * np.pi * rho * nparticles**2 * kb * T)
    # Update forces
    forces = np.zeros((nparticles, 3))
    potential_energy = compute_potential_energy_and_forces(positions, forces)
    # Update velocities
    velocities += 0.5 * forces / m1 * dt

# Compute the final viscosity estimate
viscosity /= (steps/freq)

# Print the result
print("Viscosity estimate:", viscosity, "Pa s")
