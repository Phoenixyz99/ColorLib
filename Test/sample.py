import numpy as np
import matplotlib.pyplot as plt
from ColorLib import cieobserver

plt.style.use('dark_background')

wavelengths = np.linspace(320, 780, 340)
colors = []

max_rgb_linear = 0

for wl in wavelengths:
    xyz = cieobserver.wavelength_to_xyz(wl)
    rgb_linear = cieobserver.xyz_to_rgb(xyz)

    max_rgb_linear = max(max_rgb_linear, np.max(rgb_linear))
    
    rgb_srgb = cieobserver.linear_to_srgb(rgb_linear)
    colors.append(np.clip(rgb_srgb, 0, 1))

print(max_rgb_linear)

plt.figure(figsize=(10, 2))
plt.imshow([colors], aspect='auto', extent=[380, 780, 0, 1])
plt.xlabel('Wavelength (nm)')
plt.yticks([])
plt.title('Colors (1nm intervals) vs. Wavelength')
plt.show()