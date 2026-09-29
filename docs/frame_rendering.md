# Frame Rendering

The traces of [trace generation](trace_generation.md) are the ground truth, and the renderer turns them into the streams a camera would record of the neck.
Every stream is a sequence of frames on one clock, and every frame is a function of the traces at its time and of a scene drawn once per sample.

## Frame Grid

The frames run at the sample rate $f_s$ of the config (default 30 Hz) for the nominal duration $T$ (default 30 s), so frame $j$ is at

$$t_j = j / f_s, \qquad 0 \le j < T f_s.$$

This is the same rate the traces are resampled to, so a stored trace sample and a frame share a timestamp and no interpolation is needed to align them.
The renderer itself does not read the resampled traces.
The pulse reaches each pixel of a vessel at a different time, so a frame needs the traces at times between samples, and the renderer reads the 1000 Hz traces of the padded grid directly, by linear interpolation.
The padding of 2 s at each end covers the propagation delays without running off the trace.

Three streams are delivered at this rate: colour as 8-bit RGB, near-infrared as 16-bit, and depth as 16-bit integer millimetres, so a depth value of 501 is 501 mm from the camera.

## Camera

The rig is that of the Neckflix recordings: a camera with a $90°$ horizontal field of view and $W = 3840$ px across, of which a square crop of $C = 650$ px about the neck is delivered.
By default the crop is delivered at its native 650 px; a smaller delivered size $N$, such as the 300 px of the Neckflix release, is an argument given when a dataset is generated, and the sensor stage then area-averages the crop down to it.
We model it as a pinhole with focal length

$$f = \frac{W / 2}{\tan(45°)} = 1920\ \text{px}.$$

Camera coordinates are $x$ right, $y$ down and $z$ forward, and a pixel at offset $(u, v)$ in native pixels from the crop centre looks along the ray $\mathbf{v} = (u/f,\ v/f,\ 1)$.
Rendering on a 650-pixel grid over the crop is native resolution, and the geometry below can be evaluated on that grid or on any coarser one.

The distance from the camera to the skin at the frame centre is drawn as $D \sim \mathcal{U}(500, 1000)$ mm.
The scale at that distance is $D / f$ mm per native pixel, 0.26 to 0.52 mm, or $C / N$ times that per delivered pixel when a smaller size is requested.
Because the projection is a true pinhole, points of the neck nearer the camera than the frame centre appear larger, and the depth stream and the colour streams are consistent with each other.

## Neck and Vessel Geometry

The neck is a cylinder of radius $r$.
The mid-neck circumference of the Neckflix population runs from 37.5 to 51 cm between its 5th and 95th percentiles; we round and widen this to $c \sim \mathcal{N}(45, 5^2)$ cm truncated to $[30, 60]$, so that necks outside that population are also seen, and take $r = 10c / 2\pi$ mm, 48 to 95 mm.
The axis lies in the plane perpendicular to the optical axis at distance $D + r$, so the ray through the frame centre meets the skin at exactly $D$, and its angle in the image is the subject's posture.
We take the torso and neck as aligned and the camera as level, so the neck lies at the head-up angle $\beta \sim \mathcal{U}(5°, 85°)$ above the horizontal, and a fair coin mirrors it left to right so that either side of the neck is seen, giving $\alpha = \beta$ or $180° - \beta$ and the unit direction $\mathbf{d} = (\cos\alpha, -\sin\alpha, 0)$ and across-neck direction $\mathbf{n} = (\sin\alpha, \cos\alpha, 0)$.

A pixel's ray $s\mathbf{v}$ meets the cylinder where its distance from the axis is $r$.
With $\mathbf{c} = (0, 0, D + r)$ a point on the axis and $\mathbf{v}_\perp = \mathbf{v} - (\mathbf{v}\cdot\mathbf{d})\,\mathbf{d}$ the part of the ray across the axis,

$$|s\mathbf{v}_\perp - \mathbf{c}|^2 = r^2, \qquad s = \frac{B - \sqrt{B^2 - AC}}{A}, \qquad A = |\mathbf{v}_\perp|^2,\ B = \mathbf{v}_\perp\cdot\mathbf{c},\ C = |\mathbf{c}|^2 - r^2,$$

taking the nearer root, and since the ray has unit $z$ the parameter $s$ is the depth of that pixel.
Rays with $B^2 < AC$ miss the neck; they are given the depth of the axis plane $D + r$, and what they see is left to the appearance stage.
For a skin point $\mathbf{p} = s\mathbf{v}$ we record its coordinate along the axis $\ell = \mathbf{p}\cdot\mathbf{d}$, zero at the frame centre, and its angle round the cylinder $\phi$, zero at the point facing the camera and positive towards $\mathbf{n}$.

The carotid and the jugular are straight tubes parallel to the axis.
Each has an angular position $\phi_v$ and an axis radius $\rho_v = r - h_v - D_v / 2$, where $h_v$ is the depth of its anterior wall below the skin and $D_v$ its diameter.
The midpoint of the pair sits an arc $o_\phi \sim \mathcal{U}(-25, 25)$ mm from the frame centre, so $\phi_{\mathrm{mid}} = o_\phi / r$, and the two axes are separated by an arc $\Delta \sim \mathcal{U}(10, 16)$ mm, $\phi_v = \phi_{\mathrm{mid}} \pm \Delta / 2r$.
The internal jugular lies lateral to the common carotid and overlaps it in most patients [14], and the camera may view either side of the neck, so which vessel takes the positive side is a fair coin.
The visible segment has length $L \sim \mathcal{U}(40, 80)$ mm with its midpoint at $o_\ell \sim \mathcal{U}(-15, 15)$ mm along the axis, and its caudal end, the heart end from which the pulses arrive, is at $\ell_0 = o_\ell - L / 2$.
The posture is recorded with the sample as supine below $30°$, recumbent to $60°$ and sitting above, matching the three positions of the Neckflix recordings.

| quantity | prior (mm) | source |
| --- | --- | --- |
| carotid diameter $D_a$ | $\mathcal{N}(6.3, 0.9^2)$ on $[4, 9]$ | common carotid $6.10 \pm 0.80$ in women and $6.52 \pm 0.98$ in men, $n = 500$ [11] |
| jugular diameter $D_j$ | $\mathcal{N}(11, 3.5^2)$ on $[4, 20]$ | $10.55 \pm 4.19$ across postures [12]; 14.1 supine [13] |
| jugular depth $h_j$ | $\mathcal{N}(15, 4^2)$ on $[5, 30]$ | skin to vein 15 mm supine [13] |
| carotid depth $h_a$ | $\mathcal{N}(20, 4^2)$ on $[8, 35]$ | assumed: posteromedial to and overlapped by the jugular [14] |

No reachable study reports the depth of the common carotid below the skin directly, so its prior is an assumption placed one jugular radius deeper than the jugular, to be replaced when a measurement is found.
The jugular diameter is drawn once here as a static size; its distension with pressure belongs to the pulse stage.

From the scene we compute, for every pixel, the depth, whether the ray meets the neck, $\phi$, the axial coordinate from the caudal end $\ell - \ell_0$, and the distance from the skin point to each vessel axis in the cross-section plane,

$$d_v = \sqrt{r^2 + \rho_v^2 - 2 r \rho_v \cos(\phi - \phi_v)},$$

which equals $h_v + D_v / 2$ directly over the vessel and grows to either side.
It is this distance that the later stages use to spread each vessel's effect over the skin, which is the physical reason a deep vessel shows as a broad motion rather than a sharp line.
The vessel labels are the projected lumens, the pixels within the segment whose arc from the axis $|r(\phi - \phi_v)|$ is at most $D_v / 2$, the artery written over the vein where they overlap, and they are delivered on the same grid as the frames as the masks of the sample.

## Pulse Propagation

The traces enter here.
The output of this stage is a pressure in mmHg at every pixel of the neck at every frame time, one field for the carotid and one for the jugular, and nothing is painted yet.

The carotid pressure is central, not the arm pressure the ABP trace records.
We generate it with the same Windkessel as the ABP and a carotid row of the shape table, in which the reflected wave arrives in late systole before the valve closes, so a beat has a first systolic peak, a lower second peak and then the incisura at the end of ejection:

| site | $c_s$ | $b_s$ | $a_s$ | $c_d$ | $b_d$ | $a_d$ | $z_c$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| carotid | 0.25 | 0.10 | 1 | 0.75 | 0.12 | 0.30 | 0.08 |

These are initial values, tuned by eye against the brachial and radial beats at 40, 70 and 140 bpm.
The foot of beat $k$ is at $r_k + p + \Delta_{ac}$, with the same pre-ejection draw as the ABP and an aortic-to-carotid transit $\Delta_{ac} \sim \mathcal{N}(0.043, 0.006^2)$ s truncated to $[0.02, 0.07]$, the delay from aortic valve opening to the carotid upstroke measured by Hasegawa et al. [15].
The level is the same $\mathrm{MAP}$ draw as the ABP, and the pulse pressure is the brachial draw scaled by $\kappa_c \sim \mathcal{U}(0.75, 1.0)$, the inverse of the amplification from the central arteries to the arm [16].
The respiratory swing is the ABP draw, since it is intrathoracic, and there is no measurement noise, since this is the pressure in the vessel rather than a catheter trace.
This trace is defined at the caudal end of the visible segment.

Along the segment the wave travels at the local carotid pulse wave velocity $v_a \sim \mathcal{N}(6.0, 1.5^2)$ m/s truncated to $[3, 12]$, between the values at the start and the end of systole measured by ultrafast ultrasound in healthy adults, $5.36 \pm 1.27$ and $6.99 \pm 1.93$ m/s [17].
A pixel at axial distance $\ell$ from the caudal end reads the trace delayed by $\ell / v_a$, 10 ms over a 60 mm segment at the prior centre and under 30 ms at the extremes of the priors, so the carotid pulse is close to simultaneous along the neck.

The jugular reads the central venous pressure.
Its delay behind the catheter has been measured directly: Zamboni et al. recorded the internal jugular cross-sectional area by ultrasound against a catheter CVP in 34 patients and found the area lagging the pressure by 0.241 s with a standard deviation of 0.175 s [23].
That lag is between the pressure at the atrium and the area of the vein at the neck, which is what is rendered, so it covers both the travel of the wave up the vein and the response of the wall, and we draw the delay at the caudal end of the segment as $\Delta_j \sim \mathcal{N}(0.241, 0.175^2)$ s truncated to $[0.04, 0.60]$.
Along the segment the wave travels at a venous pulse wave velocity $v_j \sim \mathcal{U}(1.5, 3.0)$ m/s.
Venous pulse wave velocity has been measured at $1.78 \pm 0.06$ m/s supine rising to $2.26 \pm 0.19$ m/s at $60°$ head-up in the femoral vein [18], and the same transit-time method has been applied to the internal jugular [19]; we take the femoral range as the prior until jugular values are published.
A pixel at $\ell$ reads the CVP delayed by $\Delta_j + \ell / v_j$, so the whole jugular lags the heart by a substantial and variable fraction of a beat, and the lag grows along the segment by up to 50 ms.

$$P_a(\mathbf{x}, t) = \mathrm{ABP}_{\mathrm{carotid}}\!\left(t - \frac{\ell(\mathbf{x})}{v_a}\right), \qquad P_j(\mathbf{x}, t) = \mathrm{CVP}\!\left(t - \Delta_j - \frac{\ell(\mathbf{x})}{v_j}\right),$$

with $\ell$ clipped to $[0, L]$ so that pixels beyond the segment ends hold the end values, and both traces read from the padded 1000 Hz grid by linear interpolation.
The CVP is read without its measurement noise, as for the carotid.

The posture $\beta$ drawn with the geometry has no effect on the pressures at this stage.
In a real neck the venous pressure at a point is the CVP less the weight of the blood column above the atrium, and the vein collapses where that falls to zero, so that the height of the pulsation depends on posture and CVP; we do not model the column or the collapse, and the whole segment pulses.

## Static Appearance

The skin of the neck is one of the ten swatches of the Monk Skin Tone scale [20], drawn uniformly, and its colour is the swatch's sRGB hex value exactly.
This is a deliberate choice: the swatches are the reference colours of the scale, while a camera's rendering of a given tone depends on its white balance and lighting, as the Neckflix skin colours by tone show.
The rig's rendering of skin is left to the illumination and sensor stages, and the physiology stage that follows changes this colour through the optical model of Jacques [21], so the swatch is the colour of the skin at rest.

The neck is lit as a Lambertian cylinder.
The outward normal at a skin point is at its angle $\phi$ round the cylinder, and the light is a directional source in the cross-section plane at an angle $\lambda \sim \mathcal{U}(-60°, 60°)$ from the camera axis, together with an ambient share $a \sim \mathcal{U}(0.3, 0.9)$:

$$S(\mathbf{x}) = a + (1 - a)\,\max\!\bigl(0, \cos(\phi(\mathbf{x}) - \lambda)\bigr),$$

so the brightest strip of skin faces the light rather than the camera, and the far side of the cylinder falls to the ambient level.

The skin carries a multiplicative texture, a Gaussian field of unit mean and standard deviation $\tau \sim \mathcal{N}(0.045, 0.015^2)$ truncated to $[0.01, 0.10]$, the Neckflix texture standard deviation of 3.5 to 9.7 levels divided by the skin level.
It is specified at the delivered resolution; when a smaller size is requested it is drawn on the crop grid with its standard deviation divided by the noise gain of the sensor stage's area average, so that the average returns the drawn value.

Beside the neck, where a ray misses the cylinder, the frame is a flat colour with each channel drawn as $\mathcal{U}(20, 200)$ levels, under the same texture as noise.
It stands in for whatever the crop shows past the neck and carries no shading or physiology.

The near-infrared stream sees the same cylinder with the same texture, but it is lit by the depth camera's own illuminator rather than the room, so its shading $S_0$ is the same Lambertian factor with the light at $\lambda = 0$, from the camera, and the same ambient share; the evidence for this is given with the pulse in Blood to Colour.
Its levels come from the Neckflix infrared stream itself, probed over the 168 recordings that carry it: five raw 16-bit frames per recording, with skin masked from the colour frames.
Skin averages 1202 to 3142 between the 5th and 95th percentiles across recordings, with a median of 2184, and the non-skin surround, bedding and chest, 1298 to 3104 with a median of 2010; the brightest pixel of a frame is typically under 5000, so the stream uses only the bottom tenth of its 16-bit range, and the level does not depend on posture.
We therefore draw the skin level $I_{\mathrm{IR}} \sim \mathcal{N}(2200, 600^2)$ truncated to $[800, 4000]$ and the background $\mathcal{N}(2000, 550^2)$ truncated to $[800, 3500]$, both in raw 16-bit units, and the quantiles are recorded in the priors file for the dataset.
The same probe gives the temporal noise of the infrared skin, a median standard deviation of 20 units, which the sensor stage will use.

The unpulsed frame is then

$$\mathrm{RGB}_0(\mathbf{x}) = \mathbf{c}_{\mathrm{Monk}}\, S(\mathbf{x})\, T(\mathbf{x}), \qquad \mathrm{IR}_0(\mathbf{x}) = I_{\mathrm{IR}}\, S_0(\mathbf{x})\, T(\mathbf{x})$$

on the neck, with the flat colours times $T$ elsewhere.

## Pressure to Blood Volume

The pressure fields become blood, and the blood moves the skin.
The output of this stage is, at every pixel and every frame time, the lift of the skin over each vessel in millimetres, which is also the thickness of the extra blood under that pixel, the depth of the lifted skin, and a uniform skin pulse; the colour of the blood belongs to the optical stage.

Each vessel's pressure changes its lumen area about the area at mean pressure, which is the size drawn with the geometry.
The carotid follows the arterial compliance measured by echo-tracking, the change in lumen area per unit change in pressure, which Uejima et al. report by age in 2,000 healthy adults as 1.36 (0.38) mm²/kPa in men and 1.23 (0.38) in women at 18 to 29 years falling to 0.82 (0.31) and 0.70 (0.29) at 60 to 74 [22]; we draw $C_a \sim \mathcal{N}(1.0, 0.35^2)$ mm²/kPa truncated to $[0.3, 2.2]$, with 1 mmHg $= 0.1333$ kPa, so that
$$\Delta A_a(\mathbf{x}, t) = C_a\,\bigl(P_a(\mathbf{x}, t) - \bar P_a\bigr),$$
where $\bar P_a$ is the mean of the carotid trace over the nominal duration.
The jugular is far more compliant.
Zamboni et al. recorded the internal jugular cross-sectional area by B-mode ultrasound beat by beat against a catheter CVP in 34 patients: the area averaged 0.98 cm² with a within-beat standard deviation of 0.05 cm² (0.008 to 0.114), while the CVP had a within-beat standard deviation of 1.21 cmH₂O (0.55 to 2.17), which is 0.89 mmHg [23].
The ratio, 5 mm² per 0.89 mmHg, is 5.7 percent of the mean area per mmHg, and we draw the jugular area strain $\varepsilon_j \sim \mathcal{N}(0.057, 0.02^2)$ per mmHg truncated to $[0.02, 0.12]$ and apply it to the drawn area $A_j = \pi D_j^2 / 4$,
$$\Delta A_j(\mathbf{x}, t) = \varepsilon_j A_j\,\bigl(P_j(\mathbf{x}, t) - \bar P_j\bigr).$$
Both relations are linear about the mean pressure; the vein's true pressure-area curve is sigmoid and flattens as it fills or empties, which we do not model, as we do not model its collapse.
The same study supplies the lag $\Delta_j$ of the propagation stage.

The tissue over a vessel is treated as an incompressible elastic half-space, and a change of lumen area $\Delta A$ along a line at depth $h$ below its surface lifts the surface by
$$u(s) = \frac{\Delta A}{\pi}\,\frac{h}{h^2 + s^2}$$
at lateral distance $s$, a Lorentzian of half-width $h$ whose integral over the surface is exactly $\Delta A$, so the blood the vessel takes in is the volume by which the skin rises.
A deep vessel therefore lifts a broad low strip and a shallow one a narrow high one, with no free parameter: the spread is the depth.
On the cylinder $h^2 + s^2$ is the squared distance $d_v^2$ from the skin point to the vessel axis of the geometry stage, and the depth is that of the axis, $h_v^c = h_v + D_v / 2$, so
$$u_v(\mathbf{x}, t) = \Delta A_v(\mathbf{x}, t)\,\frac{h_v^c}{\pi\, d_v(\mathbf{x})^2}, \qquad v \in \{a, j\}.$$
Directly over the vessel the lift is $\Delta A_v / \pi h_v^c$ and at one axis depth to the side it is half that.
At the priors' centres a carotid pulse of 40 mmHg is 5.3 mm² and lifts the skin 0.07 mm from an axis 23 mm down, while a jugular pulse of 5 mmHg on a 95 mm² lumen is 27 mm² and lifts it 0.43 mm from 20 mm down, so the jugular is the visible motion of the neck and the carotid is a faint one, as at the bedside.
The pressures beyond the ends of the segment hold the end values, so the whole length of the neck lifts with the vessels, which continue past the segment.

The lift is along the outward normal of the cylinder, and the camera sees its component towards it, so the depth of the skin at a frame time is
$$z(\mathbf{x}, t) = z_0(\mathbf{x}) - \bigl(u_a(\mathbf{x}, t) + u_j(\mathbf{x}, t)\bigr)\cos\phi(\mathbf{x}),$$
with $z_0$ the depth of the geometry stage.
The depth stream is delivered in integer millimetres, and a lift of a fraction of a millimetre is not lost to that quantisation: the value is rounded stochastically, $\lfloor z + U \rfloor$ with $U \sim \mathcal{U}(0, 1)$ drawn per pixel and frame, so that a lift of 0.3 mm moves 30 percent of the pixels over the vessel by one millimetre.
The sensor stage adds the depth camera's noise before this rounding.

The skin itself pulses with the blood in its microvasculature, uniformly over the neck.
This is the photoplethysmographic pulse of the neck skin, and it is generated as the finger PPG is, the finger row of the Windkessel with its foot $\delta_s \sim \mathcal{U}(0.02, 0.06)$ s after the carotid foot, rescaled to peak-to-trough 1 about a zero mean, then modulated by respiration with the PPG's own draws of amplitude modulation $m$ and wander $w$ and no noise,
$$b(t) = b_0(t)\,\bigl(1 - m\,x(t)\bigr) - w\,x(t),$$
so that the respiratory rate can be recovered from it.
The transit $\delta_s$ from the carotid to the skin is an assumption, to be replaced when a measurement is found.
The skin pulse is dimensionless; its scale in blood is set by the optical stage.

## Blood to Colour

The skin pulse of the previous stage becomes a change of colour, and the lift becomes a change of shading.
The vessels themselves do not colour the skin: with the optical properties below, the fluence falls by $e$ every 0.4 mm in green, 1.2 mm in red and 2 mm at 850 nm, so a vessel wall 15 mm below the skin is attenuated by more than $e^{-15}$ on the way in and out in every band, and the blood in the carotid and jugular is invisible at the depths of the geometry priors.
What the camera sees of them is the motion of the skin, through the shading of its tilt here and through the depth stream.

The dermis is a turbid medium whose absorption is that of its blood, water and a bloodless baseline [21],
$$\mu_a(\lambda) = B\,\bigl(S\,\mu_a^{\mathrm{oxy}}(\lambda) + (1 - S)\,\mu_a^{\mathrm{deoxy}}(\lambda)\bigr) + W\,\mu_a^{\mathrm{water}}(\lambda) + \mu_a^{\mathrm{base}}(\lambda),$$
with $B$ the blood volume fraction, $S$ its oxygen saturation and $W = 0.65$ the water fraction.
The blood absorption is $2.303\,\varepsilon\,c$ with the molar extinction $\varepsilon$ of oxy- and deoxy-haemoglobin from Prahl's compilation [24] and $c = 2.33$ mM for 150 g/L of haemoglobin, the water absorption is that of Hale and Querry [25], and the baseline is Jacques' skin baseline $0.244 + 85.3\,e^{-(\lambda - 154)/66.2}$ cm⁻¹ [26].
The reduced scattering of skin is $\mu_s' = 46.0\,(\lambda / 500)^{-1.421}$ cm⁻¹ [21].
The camera bands are taken as nominal centre wavelengths, 460, 540 and 610 nm for blue, green and red, and 850 nm for the depth camera's illuminator, which lights the infrared stream:

| band | $\lambda$ (nm) | $\mu_a^{\mathrm{oxy}}$ | $\mu_a^{\mathrm{deoxy}}$ | $\mu_a^{\mathrm{base}}$ | $\mu_s'$ |
| --- | --- | --- | --- | --- | --- |
| blue | 460 | 238 | 125 | 1.08 | 51.8 |
| green | 540 | 285 | 250 | 0.49 | 41.2 |
| red | 610 | 8.1 | 50.6 | 0.33 | 34.7 |
| infrared | 850 | 5.7 | 3.7 | 0.25 | 21.6 |

all in cm⁻¹, the blood values for whole blood.
The diffuse reflectance of the skin in a band is that of a semi-infinite medium in the diffusion approximation [27], with the transport albedo $a' = \mu_s' / (\mu_a + \mu_s')$ and the internal reflection parameter $A = 3.25$ for a refractive index of 1.4,
$$R = \frac{a'}{2}\,\Bigl(1 + e^{-\frac{4}{3} A \sqrt{3(1 - a')}}\Bigr)\,e^{-\sqrt{3(1 - a')}}.$$

The resting blood fraction is drawn as $B_0 \sim \mathcal{N}(0.015, 0.008^2)$ truncated to $[0.002, 0.05]$, within the 0.2 to 7 percent of dermis [21], at a mixed arterial and venous saturation $S_0 \sim \mathcal{U}(0.6, 0.9)$, which is an assumption.
The skin pulse $b(t)$ adds arterial blood at $S = 0.98$, a peak-to-trough fraction $\Delta B$ of the tissue, and the colour of the skin at time $t$ in band $k$ is the resting colour times the ratio of reflectances,
$$\Pi_k(t) = \frac{R_k\bigl(B_0, S_0;\ \Delta B\, b(t)\bigr)}{R_k(B_0, S_0)},$$
uniform over the neck.
Only the ratio enters, so the melanin of the epidermis, a filter common to numerator and denominator, cancels, and the resting colour stays the Monk swatch; a darker tone pulses by the same fraction of a lower level, which is how the signal-to-noise of the pulse falls with tone in a camera.

The size of the pulse is set from the Neckflix colour stream itself.
Over 312 recordings, 20 s of each channel was averaged over the neck skin frame by frame and band-passed about the recorded heart rate; in the 130 recordings whose green spectrum has a clear cardiac peak the peak-to-trough pulse of the green level is 0.13 percent of the level at the median and 0.07 to 0.29 between the 5th and 95th percentiles, with no trend across Monk tones 2 to 6 or across the three postures, and the quantiles are recorded in the priors file.
We therefore draw the green pulse $g \sim \mathcal{N}(0.0015, 0.0007^2)$ truncated to $[0.0005, 0.004]$ and set $\Delta B = g / |\partial \ln R_g / \partial B|$ at the drawn resting state, so that the rendered green pulses by the drawn fraction whatever the resting blood, which changes the sensitivity from 52 per unit blood at $B_0 = 0.005$ to 17 at $B_0 = 0.03$.
At the prior centres $\Delta B$ is about $5 \times 10^{-5}$, one three-hundredth of the resting blood.

The model then fixes the other bands, and the same probe checks them.
Relative to green, the model gives a red pulse of 0.14, blue 0.84 and infrared 0.15 at $B_0 = 0.015$; the recordings give 0.21 (0.15 to 0.31 between quartiles) for red, 0.53 (0.38 to 0.74) for blue and 0.63 (0.51 to 0.74) for infrared.
Red agrees.
Blue is high in the model, which treats the dermis as homogeneous while blue light is stopped in its upper, less vascular part, and we keep the model rather than tune the wavelength.
The infrared disagrees by four times, and the reason is that the infrared stream is not a passive image: it is lit by the depth camera's own illuminator, whose return depends on the range and slope of the skin, so the cardiac signal in that stream is dominated by the motion of the skin over the vessels rather than by blood.
The infrared is therefore lit from the camera, the shading $S_0$ of the appearance stage with the light at $0°$ and the ambient share drawn as for the colour, and the infrared pulse of blood is kept at its optical value.

The lift $u$ of the pressure stage tilts the skin.
The surface $u(s)$ along the arc $s = r\phi$ turns the outward normal by $-\partial u / \partial s$, which for the Lorentzian lift is
$$\theta_v(\mathbf{x}, t) = \Delta A_v(\mathbf{x}, t)\,\frac{2 h_v^c \rho_v \sin(\phi - \phi_v)}{\pi\, d_v(\mathbf{x})^4},$$
zero on the crest over the vessel and of opposite sign on its two flanks, and the shading of the appearance stage is evaluated at $\phi + \theta_a + \theta_j$,
$$S(\mathbf{x}, t) = a + (1 - a)\,\max\!\bigl(0, \cos(\phi(\mathbf{x}) + \theta_a(\mathbf{x}, t) + \theta_j(\mathbf{x}, t) - \lambda)\bigr).$$
A jugular pulse of 5 mmHg on a 95 mm² lumen 20 mm down tilts the skin by up to 0.01 radians, which under a light at $60°$ changes the shading by about 1 percent, brighter on the flank turned towards the light and darker on the other: a band of the same order as the skin pulse, and the mechanism by which the jugular pulse is seen in a colour video under tangential light.

The frame at time $t$ is then
$$\mathrm{RGB}(\mathbf{x}, t) = \mathbf{c}_{\mathrm{Monk}}\,\boldsymbol{\Pi}(t)\, S(\mathbf{x}, t)\, T(\mathbf{x}), \qquad \mathrm{IR}(\mathbf{x}, t) = I_{\mathrm{IR}}\,\Pi_{\mathrm{ir}}(t)\, S_0(\mathbf{x}, t)\, T(\mathbf{x}),$$
on the neck, with $S_0$ the shading lit from the camera, the flat colours times $T$ beside it, and the depth $z(\mathbf{x}, t)$ of the pressure stage; these are continuous values on the render grid, and the sensor stage samples, blurs, adds noise and quantises them.

## Sensor

The continuous frame becomes what the camera records, in four steps, each frame independently.

The lens and the demosaic blur the image: a Gaussian of width $\sigma_b \sim \mathcal{U}(0.5, 1.5)$ native pixels is applied to the three streams on the 650-pixel grid.
This is an assumed prior; the neck has no sharp feature but its silhouette, so the blur mostly sets how crisp the texture is, and the depth stream, which in the rig is registered to the colour camera from a coarser sensor, gets the same blur for want of a better model.

The crop is then delivered at its native 650 pixels or, when a smaller size $N$ is requested, area-averaged to it: each delivered pixel is the mean of the native pixels it covers, weighted by the overlap, which is exact for any ratio.
The average reduces white noise by a gain that is $N / C$ for an integer ratio and somewhat more otherwise, since a delivered pixel then spreads over parts of three native pixels; the texture of the appearance stage is drawn with its standard deviation divided by that gain, so the delivered texture has the drawn value at any size.
The vessel labels and the neck mask are resampled with the frames by the majority of the area each delivered pixel covers.

Noise is added at the delivered size, Gaussian per pixel and frame, with an independent standard deviation for each stream: $\mathcal{U}(0.5, 2)$ levels per channel for the colour, $\mathcal{N}(20, 5^2)$ units truncated to $[8, 40]$ for the infrared, the temporal standard deviation of skin measured in the Neckflix infrared stream, and $\mathcal{U}(0.5, 2)$ mm for the depth.
The colour and depth values are assumed; the noise is not made to depend on the level, which at 8 bits is invisible, and no mains flicker is added.

The streams are then quantised to their delivered types.
Colour is rounded and clipped to 8 bits and the infrared to 16.
Depth takes its noise and is then rounded stochastically to integer millimetres as described with the lift, so that the jugular's fraction of a millimetre survives quantisation as the fraction of pixels that move, and the mean depth over the vessel label in the delivered frames follows the CVP.

## Outputs

A sample is drawn once from one seed, in a fixed order, so the seed reproduces it: the trace priors and the traces of [trace generation](trace_generation.md), the propagation and distension priors and the carotid pressure and skin pulse they set, then the camera, the scene, the appearance, the optics and the sensor, then the texture, and then the fields above.
The sensor noise continues the same stream, a batch of frames at a time, so the frames are reproducible too; on a GPU the noise is drawn by the device's own generator seeded from that stream, so a seed reproduces its frames there as well, though the CPU and GPU realisations of the noise differ.
The frames are stepped on the clock of the Frame Grid in batches, each batch rendered on the native crop from the padded 1 kHz traces in float32, recorded by the sensor at the delivered size, and written as one zarr store per sample in the layout of remote-physiology's cache contract: the colour, infrared and depth videos as $(C, T, H, W)$ arrays of their delivered types with per-frame timestamps, the five traces of the trace document cropped to $[0, T)$ and resampled to the frame rate so that sample $j$ of a trace and frame $j$ share a timestamp, the vessel labels and the neck mask at the delivered size, and the posture, the catheter site, the Monk tone, the seed and every drawn value as attributes.
The stored arterial pressure is the catheter trace at the drawn site, brachial or radial, with its noise, and the stored CVP has its catheter noise, while the frames were rendered from the carotid pressure and the noiseless CVP, since a camera sees the vessel and not the catheter.
No sample is redrawn or rejected: the jugular lifts the skin in every sample by an amount set by its CVP and compliance, and that amount is recorded with the sample.

## References

[11] J. Krejza, M. Arkuszewski, S. E. Kasner, J. Weigele, A. Ustymowicz, R. W. Hurst, B. L. Cucchiara and S. R. Messé, "Carotid artery diameter in men and women and the relation to body and neck size," *Stroke*, vol. 37, no. 4, pp. 1103–1105, 2006. doi:10.1161/01.STR.0000206440.48756.f7.

[12] S. L. Solanki, J. R. Doctor, S. J. Kapila, A. Jain, M. Joshi and V. P. Patil, "Ultrasonographic assessment of internal jugular vein diameter and its relationship with the carotid artery at the apex, middle, and base of the triangle formed by two heads of sternocleidomastoid muscle: a pilot study in healthy volunteers," *Saudi Journal of Anaesthesia*, vol. 12, no. 4, pp. 578–583, 2018. doi:10.4103/sja.SJA_309_18.

[13] A. N. Sibai, E. Loutfi, M. Itani and A. Baraka, "Ultrasound evaluation of the anatomical characteristics of the internal jugular vein and carotid artery: facilitation of internal jugular vein cannulation," *Middle East Journal of Anaesthesiology*, vol. 19, no. 6, pp. 1305–1320, 2008.

[14] T. Maecken, C. Marcon, S. Bomas, M. Zenz and T. Grau, "Relationship of the internal jugular vein to the common carotid artery: implications for ultrasound-guided vascular access," *European Journal of Anaesthesiology*, vol. 28, no. 5, pp. 351–355, 2011. doi:10.1097/EJA.0b013e328341a492.

[15] M. Hasegawa, D. Rodbard and Y. Kinoshita, "Timing of the carotid arterial sounds in normal adult men: measurement of left ventricular ejection, pre-ejection period and pulse transmission time," *Cardiology*, vol. 78, no. 2, pp. 138–149, 1991. doi:10.1159/000174778.

[16] A. P. Avolio, L. M. Van Bortel, P. Boutouyrie, J. R. Cockcroft, C. M. McEniery, A. D. Protogerou, M. J. Roman, M. E. Safar, P. Segers and H. Smulyan, "Role of pulse pressure amplification in arterial hypertension: experts' opinion and review of the data," *Hypertension*, vol. 54, no. 2, pp. 375–383, 2009. doi:10.1161/HYPERTENSIONAHA.109.134379.

[17] W. Yang, Y. Wang, Y. Yu, L. Mu, F. Kong, J. Yang, D. Jia and C. Ma, "Establishing normal reference value of carotid ultrafast pulse wave velocity and evaluating changes on coronary slow flow," *International Journal of Cardiovascular Imaging*, vol. 36, no. 10, pp. 1931–1939, 2020. doi:10.1007/s10554-020-01908-3.

[18] L. Ermini, C. Ferraresi, C. De Benedictis and S. Roatta, "Objective assessment of venous pulse wave velocity in healthy humans," *Ultrasound in Medicine & Biology*, vol. 46, no. 3, pp. 849–854, 2020. doi:10.1016/j.ultrasmedbio.2019.11.003.

[19] N. R. George, R. Manoj, R. K. V, N. P. M, M. Sivaprakasam and J. Joseph, "Ultrasound for venous local pulse wave velocity: comparison of pulse transit time methods," in *Proc. 45th Annual International Conference of the IEEE Engineering in Medicine and Biology Society*, 2023, pp. 1–4. doi:10.1109/EMBC40787.2023.10340269.

[20] E. Monk, "The Monk Skin Tone Scale," *SocArXiv*, 2023. doi:10.31235/osf.io/pdf4c. Swatches: [skintone.google](https://skintone.google/get-started).

[21] S. L. Jacques, "Optical properties of biological tissues: a review," *Physics in Medicine and Biology*, vol. 58, no. 11, pp. R37–R61, 2013. doi:10.1088/0031-9155/58/11/R37.

[22] T. Uejima, F. D. Dunstan, E. Arbustini, K. Łoboz-Grudzień, A. D. Hughes, S. Carerj, V. Favalli, F. Antonini-Canterin, O. Vriz, D. Vinereanu, J. L. Zamorano, B. A. Popescu, A. Evangelista, P. Lancellotti, G. Lefthériotis, M. Kozakova, C. Palombo and A. G. Fraser, for the E-Tracking International Collaboration Group, "Age-specific reference values for carotid arterial stiffness estimated by ultrasonic wall tracking," *Journal of Human Hypertension*, vol. 34, no. 3, pp. 214–222, 2020. doi:10.1038/s41371-019-0228-5.

[23] P. Zamboni, A. M. Malagoni, E. Menegatti, R. Ragazzi, V. Tavoni, M. Tessari and C. B. Beggs, "Central venous pressure estimation from ultrasound assessment of the jugular venous pulse," *PLoS ONE*, vol. 15, no. 10, e0240057, 2020. doi:10.1371/journal.pone.0240057.

[24] S. A. Prahl, "Optical absorption of hemoglobin," Oregon Medical Laser Center, 1999. [omlc.org/spectra/hemoglobin](https://omlc.org/spectra/hemoglobin/).

[25] G. M. Hale and M. R. Querry, "Optical constants of water in the 200-nm to 200-μm wavelength region," *Applied Optics*, vol. 12, no. 3, pp. 555–563, 1973. doi:10.1364/AO.12.000555.

[26] S. L. Jacques, "Skin optics," *Oregon Medical Laser Center News*, 1998. [omlc.org/news/jan98/skinoptics.html](https://omlc.org/news/jan98/skinoptics.html).

[27] S. L. Jacques, "Light distributions from point, line and plane sources for photochemical reactions and fluorescence in turbid biological tissues," *Photochemistry and Photobiology*, vol. 67, no. 1, pp. 23–32, 1998. doi:10.1111/j.1751-1097.1998.tb05161.x.
