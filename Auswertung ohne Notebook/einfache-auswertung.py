import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches
import sys
from matplotlib import cm

# erstes Argument ist Dateipfad eingabebild, muss aber im gleichen ordner liegen
image_url = sys.argv[1]
image = plt.imread(image_url)[:,:,0]

height = image.shape[0]
width = image.shape[1]

#plt.figure("Red channel", figsize=(10,7))
# plt.imshow(image, cmap='gray')
# plt.show()

# FFT zur Auswahl Sattelit
fourier = np.fft.fftshift(np.fft.fft2(image))
magnitude_spectrum = np.log(np.abs(fourier+1e-9))


plt.imshow(magnitude_spectrum)
plt.show()

# jetzt wirds hässlich

rect_center_x = int(input('X-Koordinate: '))

rect_center_y = int(input('Y-Koordinate: '))

rect_center_radius = 80


# FFT

cropped_fourier = np.zeros([height, width], dtype=complex)
satellite = fourier[rect_center_y - rect_center_radius:rect_center_y + rect_center_radius,
            rect_center_x - rect_center_radius:rect_center_x + rect_center_radius]
cropped_fourier[height//2 - rect_center_radius:height//2 + rect_center_radius,
                width//2 - rect_center_radius:width//2 + rect_center_radius] = satellite
phaseangle = np.angle(np.fft.ifft2(np.fft.fftshift(cropped_fourier)))




# phase unwrap
x2 = np.arange(-height, height)
y2 = np.arange(-width, width)
x2 = x2**2
y2 = y2**2
x2 = np.roll(x2, height)
y2 = np.roll(y2, width)
r2s = np.add.outer(x2, y2)
r2s = r2s.astype(np.float32)
r2s += 1e-10 # bitte keine Null
mirrored = np.zeros([height*2, width*2])
print(mirrored.shape, height, width)
print(phaseangle.shape)
mirrored[:height, :width] = phaseangle
mirrored[height:, :width] = phaseangle[::-1,:]
mirrored[height:, width:] = phaseangle[::-1, ::-1]
mirrored[:height, width:] = phaseangle[:, ::-1]
holo_cos = np.cos(mirrored)
holo_sin = np.sin(mirrored)

dt = np.fft.fft2
idt = np.fft.ifft2

phi_prime = r2s**-1.0*dt(holo_cos*idt(r2s*dt(holo_sin)) - holo_sin*idt(r2s*dt(holo_cos)))
phi_prime = idt(phi_prime)

phi_ = phi_prime.real[:height, :width]
unwrapped_phase = (phaseangle + 2*np.pi*np.round((phi_ - phaseangle) / 2 / np.pi))



# Anzeigen der Graphen
plt.imsave(fname='unwrapped_phase.png', arr=unwrapped_phase, format='png', cmap='gray')

fig1 = plt.figure()
plt.subplot(221)
plt.title("Bild")
plt.imshow(image, cmap='gray')

ax1 = plt.subplot(222)
plt.title("Fourier")
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(223)
plt.title("Phaseangle")
plt.imshow(phaseangle, cmap='gray')

plt.subplot(224)
plt.title("unwrapped phase")
plt.imshow(unwrapped_phase, cmap='gray')
plt.show()

satellite_patch = matplotlib.patches.Rectangle((rect_center_x - rect_center_radius, rect_center_y - rect_center_radius),
                                              rect_center_radius*2, rect_center_radius*2)
satellite_patch.fill = False

ax1.add_patch(satellite_patch)

