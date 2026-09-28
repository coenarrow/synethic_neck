# Trace Generation

We define a time grid at 1000 Hz spanning the nominal duration $T$ (default 30 s), padded by 2 s at each end so that beats and breaths straddling the edges are complete:

$$t_i = -2 + i\,\Delta t, \qquad \Delta t = 10^{-3}\ \text{s}, \qquad t_i < T + 2.$$

This grid is used for all traces. Once generated, each trace is cropped back to $0 \le t < T$ and resampled to the sample rate given in the config.

## Respiratory Waveform

Our first step is generation of the instantaneous respiratory rate $f(t)$ in breaths per minute.
We draw a mean rate $\bar f \sim \mathcal{U}(10, 30)$ and add slow drift following an Ornstein-Uhlenbeck process $d(t)$ with two parameters: its standard deviation $\sigma_d \sim \mathcal{U}(0, 0.15)$, expressed as a fraction of the mean rate, and its correlation time $\tau \sim \mathcal{U}(20, 60)$ s.
On the grid the process is

$$d_i = a\, d_{i-1} + \sigma_d \sqrt{1 - a^2}\; \epsilon_i, \qquad a = e^{-\Delta t / \tau}, \qquad \epsilon_i \sim \mathcal{N}(0, 1),$$

with $d_0 \sim \mathcal{N}(0, \sigma_d^2)$ so the drift starts at a typical value rather than at zero.
The rate is then

$$f(t_i) = \bar f \,\bigl(1 + \operatorname{clip}(d_i, -\tfrac12, \tfrac12)\bigr),$$

so it never leaves $[\tfrac12 \bar f, \tfrac32 \bar f]$ and in particular stays positive.
The resulting rate wanders slowly but is not smooth; it carries fine roughness from sample to sample.

The respiratory waveform is obtained by integrating the rate over time to give a phase, adding a random starting phase $\phi_0 \sim \mathcal{U}(0, 2\pi)$, and taking the sine:

$$x(t) = \sin\!\left( \frac{2\pi}{60} \int_{t_0}^{t} f(u)\, du + \phi_0 \right).$$

A value of $x = +1$ is end-inspiration.
Because the phase is the integral of the rate, a drifting rate produces smoothly varying breath lengths, and $x(t)$ becomes the shared clock for every trace that follows.

## ECG Waveform

To generate the ECG waveform, we begin with using a monte-carlo method to generate peak times of the ECG waveform.
The mean heart rate is drawn as $\bar h \sim \mathcal{U}(40, 140)$ bpm, giving the mean RR interval $\overline{RR} = 60 / \bar h$ s.
The first R wave is placed at a random point within one mean RR interval of the start of the padded grid:

$$r_0 = t_0 + u, \qquad u \sim \mathcal{U}(0, \overline{RR}).$$

Each subsequent R wave follows from the previous one by a perturbed RR interval with three terms: Gaussian jitter, respiratory sinus arrhythmia (RSA) read from the respiratory waveform at the previous R wave, and a slow Mayer wave:

$$r_{k+1} = r_k + \overline{RR}\, \max\!\Bigl( 1 + \sigma_h \epsilon_k \;-\; \rho\, x(r_k) \;+\; m \sin(2\pi \cdot 0.1\, r_k + \psi),\ 0.3 \Bigr),$$

with jitter $\sigma_h \sim \mathcal{U}(0.01, 0.06)$,
$\epsilon_k \sim \mathcal{N}(0, 1)$,
RSA fraction $\rho \sim \mathcal{U}(0.02, 0.10)$,
Mayer amplitude $m \sim \mathcal{U}(0, 0.04)$ and
Mayer phase $\psi \sim \mathcal{U}(0, 2\pi)$.

The sign of the RSA term shortens intervals that begin near end-inspiration, and because it reads $x$ at the previous beat it lags the breath by up to one interval, as in physiology.
The floor of $0.3\,\overline{RR}$ prevents any interval collapsing.
The walk continues until it passes the end of the padded grid.

We now go from the series of R-wave times $\{r_k\}$ to the ECG trace itself, in four steps.

First, every grid sample is assigned to the beat that contains it and given an angle within that beat, so that the R peak is at angle zero and one beat is one revolution:

$$k(t) = \max\{k : r_k \le t\}, \qquad \theta(t) = 2\pi\,\frac{t - r_k}{r_{k+1} - r_k}.$$

Because each beat is scaled by its own RR interval, the waves stretch and compress with every interval.

Second, the beat shape is evaluated at that angle.
This is the morphology model of ECGSYN [1]: five Gaussians in angle, one per wave, each with a centre $\theta_i$, an amplitude $a_i$ and a width $b_i$:

$$z(\theta) = \sum_{i \in \{P,Q,R,S,T\}} a_i\, b_i^2\, \exp\!\left(-\frac{\Delta_i^2}{2 b_i^2}\right), \qquad \Delta_i = \operatorname{wrap}(\theta - \theta_i) \in (-\pi, \pi].$$

| wave | $\theta_i$ | $a_i$ | $b_i$ (rad) |
| --- | --- | --- | --- |
| P | $-70°$ | 1.2 | 0.25 |
| Q | $-15°$ | $-5.0$ | 0.10 |
| R | $0°$ | 30 | 0.10 |
| S | $+15°$ | $-7.5$ | 0.10 |
| T | $+100°$ | 0.75 | 0.40 |

These are the published ECGSYN defaults.
ECGSYN integrates an ordinary differential equation whose $z$ term is $-\sum_i a_i \Delta_i \exp(-\Delta_i^2 / 2b_i^2)$; the expression above is the exact integral of that term around one revolution with the baseline relaxation dropped, so we evaluate the waveform in closed form rather than integrating.
Following ECGSYN, the widths and wave positions grow in angle with heart rate as $\eta = \sqrt{\bar h / 60}$: every $b_i$ is multiplied by $\eta$, the Q and S centres by $\eta$, and the P and T centres by $\sqrt{\eta}$.
Because a beat is a shorter time at high rates, this moves the waves earlier in time but less than in proportion to the RR interval, in the manner of Bazett's correction; scaling by the RR interval alone would move the T wave too far.

Third, the trace is scaled so the R peak is the drawn amplitude $A \sim \mathcal{U}(0.6, 1.6)$.
The ECG is in arbitrary units, labelled millivolts in the store since this is the range of an R wave in a limb lead; $A$ sets the scale and every other ECG amplitude is expressed in the same units:

$$\mathrm{ECG}_0(t) = A\, \frac{z(\theta(t))}{\max z}.$$

Fourth, the respiratory effects and noise are added, all reading the same waveform $x(t)$:

$$\mathrm{ECG}(t) = \mathrm{ECG}_0(t)\,\bigl(1 + \alpha\, x(t)\bigr) + w\, x(t) + \sigma_n\, \epsilon(t),$$

with amplitude modulation $\alpha \sim \mathcal{U}(0.02, 0.10)$ (the electrical axis shifts with lung volume), baseline wander $w \sim \mathcal{U}(0, 0.10)$, and white noise $\sigma_n \sim \mathcal{U}(0.005, 0.03)$, $\epsilon(t) \sim \mathcal{N}(0, 1)$.
The trace is then cropped to the nominal window and resampled to the config rate.

The model is one lead in sinus rhythm.
It has no QT dependence beyond the width scaling, no T-wave alternans, no ectopic beats and no lead geometry; each could be added as a separate term.

## ABP Waveform

The arterial pressure wave is mechanical, so its features are the valve events of the cardiac cycle rather than the ECG waves themselves.
On a Wiggers diagram the aortic valve opens shortly after the QRS, at the foot of the pressure upstroke, and closes at the end of the T wave, at the dicrotic notch.
We therefore take two landmarks per beat from the ECG we have already built, and place the pressure wave between them.

The aortic valve opens a pre-ejection period $p \sim \mathcal{U}(0.05, 0.10)$ s after the R wave, consistent with the systolic time intervals of Weissler et al. [2].
The T wave in our beat is a Gaussian at angle $\sqrt{\eta}\,\theta_T$ with width $\eta\, b_T$, so we take its end as two widths past the centre and convert to time with the RR interval of that beat:

$$\theta_{\mathrm{end}} = \sqrt{\eta}\,\theta_T + 2\,\eta\, b_T, \qquad t^{\mathrm{end}}_k = r_k + \frac{\theta_{\mathrm{end}}}{2\pi}\,(r_{k+1} - r_k).$$

The ejection time of beat $k$ is then the interval between the two valve events:

$$T_{s,k} = t^{\mathrm{end}}_k - r_k - p.$$

Because each beat converts angle to time with its own RR interval, the ejection time varies from beat to beat with RSA and jitter, and the ABP stays consistent with the ECG that generated it.

The wave is delayed by transit along the arterial tree.
We draw an aortic-to-brachial transit $\Delta_{ab} \sim \mathcal{U}(0.06, 0.12)$ s and a brachial-to-radial transit $\Delta_{br} \sim \mathcal{U}(0.04, 0.10)$ s.
The foot of beat $k$ is at $f_k = r_k + p + \Delta_{ab}$ at the brachial site and $f_k + \Delta_{br}$ at the radial site, with the notch shifted by the same amount.

We model the artery as a three-element Windkessel [3]: a characteristic impedance $z_c$ in series with a compliance that is filled by the ejection flow and drained through the peripheral resistance with time constant $\tau_d = 1.5$ s.
The ejection flow of each beat is two Gaussians in time since the foot, a systolic ejection and a smaller dicrotic wave, with centres and widths expressed as fractions of the ejection time so that systole compresses with heart rate while diastole does not:

$$q(t) = \sum_k \; a_s \exp\!\left(-\frac{(t - f_k - c_s T_{s,k})^2}{2 (b_s T_{s,k})^2}\right) + a_d \exp\!\left(-\frac{(t - f_k - c_d T_{s,k})^2}{2 (b_d T_{s,k})^2}\right).$$

The pressure shape is the flow across the impedance plus the compliance pressure, which is the flow filtered by the run-off:

$$s(t) = z_c\, q(t) + s_c(t), \qquad \frac{ds_c}{dt} = -\frac{s_c}{\tau_d} + q(t), \qquad s_c(t) = \int_{-\infty}^{t} q(u)\, e^{-(t-u)/\tau_d}\, du.$$

On the grid the compliance pressure is $s_{c,i} = e^{-\Delta t / \tau_d} s_{c,i-1} + \Delta t\, q_i$.
To start it in steady state, the grid is extended backwards by $5\tau_d$ with virtual beats at the first interval, and that run-in is discarded.
The impedance term makes the pressure follow the flow, giving a sharp systolic peak that falls to the notch as ejection ends; the compliance term carries the slow rise and the exponential diastolic decay until the next foot.
Without the impedance term the peak-to-notch fall would lie on the same exponential as diastole and the wave would be a sawtooth.
A long RR interval runs down to a lower diastolic pressure than a short one, as in physiology.

| site | $c_s$ | $b_s$ | $a_s$ | $c_d$ | $b_d$ | $a_d$ | $z_c$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brachial | 0.25 | 0.10 | 1 | 1.15 | 0.08 | 0.25 | 0.05 |
| radial | 0.22 | 0.08 | 1 | 1.30 | 0.10 | 0.30 | 0.08 |

The radial ejection is narrower and its impedance higher, giving a sharper and taller systolic peak, and its dicrotic wave is later and more prominent, as the pulse travels distally.
These are initial values, tuned by eye against single-beat plots.

The pressure level is set by two draws.
The mean arterial pressure $\mathrm{MAP} \sim \mathcal{N}(90, 20^2)$ mmHg and the pulse pressure $\mathrm{PP} \sim \mathcal{N}(55, 22^2)$ mmHg are drawn as normals, truncated below at 40 and 20 mmHg respectively.
These were chosen so that the systolic, diastolic and mean pressure distributions each meet the bins required by ISO 81060-3 [4] with margin over the heart-rate prior, and the mean pressure was drawn directly because its tails are the binding constraint.

The brachial shape $s(t)$ is rescaled over the nominal window to $[0, 1]$ and its mean $\bar s$ taken, then

$$\mathrm{ABP}_0(t) = \mathrm{MAP} + \mathrm{PP}\,\bigl(s(t) - \bar s\bigr),$$

so the drawn mean is exactly the time-mean of the trace, and the systolic and diastolic pressures follow from the shape as $\mathrm{MAP} + (1 - \bar s)\,\mathrm{PP}$ and $\mathrm{MAP} - \bar s\,\mathrm{PP}$.
The form factor $\bar s$ is not fixed at a third; in this model it runs from about 0.44 at 40 bpm to 0.33 at 140 bpm at the brachial site, falling with heart rate as the impedance peak becomes a larger share of the pulse.

The radial trace uses the radial shape with its own $\bar s$, feet delayed by $\Delta_{br}$, and the pulse pressure multiplied by an amplification $\kappa \sim \mathcal{U}(1.05, 1.35)$.
The mean pressure is left unchanged, as it falls by only a few mmHg along the arm.
One site is drawn per sample by a fair coin, and the trace at that site is the ABP that is stored, with the site recorded as the catheter site of the sample.

Finally the respiratory swing and noise are added to both sites, reading the same waveform $x(t)$ as every other trace:

$$\mathrm{ABP}(t) = \mathrm{ABP}_0(t) - w_r\, x(t) + \sigma\, \epsilon(t),$$

with $w_r \sim \mathcal{U}(2, 8)$ mmHg so that pressure falls at end-inspiration as in spontaneous breathing, and $\sigma \sim \mathcal{U}(0.2, 1.0)$ mmHg.
The traces are then cropped and resampled to the config rate.

The model has no wave reflection beyond the dicrotic term, no pressure dependence of the transit times, and no pulse pressure variation with respiration beyond the common swing; each could be added as a separate term.

## CVP Waveform

The central venous pressure is the right atrial pressure as a catheter in the right atrium or vena cava records it, and it is the ground truth for what the neck shows.
Each beat has three positive waves and two descents, and each is tied to an event of the cardiac cycle [5].
The a wave is atrial contraction and follows the P wave.
The c wave is the tricuspid valve bulging into the atrium as the ventricle starts to contract.
The x descent is atrial relaxation together with the tricuspid annulus being pulled down during ejection.
The v wave is passive filling of the atrium against the closed valve, and it peaks when the tricuspid valve opens, shortly after the end of the T wave.
The y descent is the rapid emptying of the atrium into the ventricle once the valve is open.

Lumped models of the circulation reproduce the a and v waves with a time-varying atrial elastance, and the x descent once the motion of the annulus is included [6], but they need the whole right heart to do so, and they cannot produce the c wave, which comes from wave propagation [7].
We therefore model the shape kinematically, as the ECG beat is, as a sum of Gaussians in time, each placed by a landmark read from the ECG we have already built.

The P wave of our beat is a Gaussian at angle $\sqrt{\eta}\,\theta_P$, so its centre in time is

$$t^{P}_k = r_k + \frac{\sqrt{\eta}\,\theta_P}{2\pi}\,(r_k - r_{k-1}),$$

and the a wave peaks an electromechanical delay $\delta_a \sim \mathcal{U}(0.10, 0.16)$ s later, which places it at about the Q wave.
Ejection starts at $f_k = r_k + p$, with the same pre-ejection draw as the ABP and no transit since the catheter sits at the heart, and ends at $t^{\mathrm{end}}_k$, so $T_{s,k}$ is as before.
The c wave sits at the start of ejection, the x descent bottoms partway through it, and the v wave peaks past its end by the isovolumic relaxation, so the c, x and v terms are placed and widened in units of $T_{s,k}$.
The a wave and the y descent are given fixed widths in seconds, since atrial contraction and rapid filling do not shorten much with heart rate, and the y descent is placed a fixed time after the v wave.

$$s(t) = \sum_k \; G(t;\, t^P_k + \delta_a,\, b_a) + h_c\, G(t;\, f_k,\, b_c T_{s,k}) - h_x\, G(t;\, f_k + c_x T_{s,k},\, b_x T_{s,k}) + h_v\, G(t;\, t^v_k,\, b_v T_{s,k}) - h_y\, G(t;\, t^v_k + d_y,\, b_y),$$

with $G(t; c, b) = \exp\!\left(-(t - c)^2 / 2b^2\right)$ and $t^v_k = f_k + c_v T_{s,k}$.

| wave | centre | width |
| --- | --- | --- |
| a | $t^P_k + \delta_a$ | $b_a = 0.05$ s |
| c | $f_k$ | $b_c = 0.06\,T_{s,k}$ |
| x | $f_k + 0.60\,T_{s,k}$ | $b_x = 0.20\,T_{s,k}$ |
| v | $f_k + 1.20\,T_{s,k}$ | $b_v = 0.18\,T_{s,k}$ |
| y | $t^v_k + 0.15$ s | $b_y = 0.07$ s |

These are initial values, tuned by eye against single-beat plots at 40, 70 and 140 bpm.
The a wave has unit height and the others are drawn as ratios to it: $h_c \sim \mathcal{U}(0, 0.3)$ since the c wave is often absent, $h_v \sim \mathcal{U}(0.5, 0.9)$ so the a wave dominates as it does in sinus rhythm, and $h_x, h_y \sim \mathcal{U}(0.5, 1.0)$.
With the c wave close to the a wave it appears as a shoulder on the descending limb rather than a separate peak.
At high heart rates the y descent runs into the next a wave and the a and v waves fuse, as they do in physiology.

The level is set by two draws, the mean $\overline{\mathrm{CVP}} \sim \mathcal{N}(7, 5^2)$ mmHg truncated to $[2, 20]$ and the peak-to-trough pulse $\mathrm{PP}_v \sim \mathcal{U}(2, 8)$ mmHg.
The shape is rescaled over the nominal window to $[0, 1]$ and its mean $\bar s$ taken, then

$$\mathrm{CVP}_0(t) = \overline{\mathrm{CVP}} + \mathrm{PP}_v\,\bigl(s(t) - \bar s\bigr),$$

so that, as for the ABP, the drawn mean is the time-mean of the trace.
The form factor $\bar s$ is about 0.4 up to 70 bpm and rises to about 0.5 at 140 bpm as the fused waves approach a sinusoid.

Finally the respiratory swing and noise are added, reading the same $x(t)$ as every other trace:

$$\mathrm{CVP}(t) = \mathrm{CVP}_0(t) - w_r\, x(t) + \sigma\, \epsilon(t),$$

with $w_r \sim \mathcal{U}(0.3, 2.5)$ mmHg and $\sigma \sim \mathcal{U}(0.1, 0.3)$ mmHg.
The swing is a far larger share of the signal than it is for the ABP, and at the lowest means the trace dips briefly below zero at end-inspiration, as a spontaneously breathing patient's does.

The model has no augmentation of the descents by inspiratory venous return, no cannon a waves from atrioventricular dissociation, and no fused c-v wave from tricuspid regurgitation; each would be a separate term.

## PPG Waveform

The photoplethysmogram we generate is the finger pulse oximeter pleth, drawn systole up as a monitor displays it.
The PPG measures the change in blood volume of the vascular bed under the sensor, and at the finger that volume pulse follows the arterial pressure pulse through the compliance of the bed [8].
Its upstroke, systolic peak, dicrotic notch and run-off are those of the pressure wave, but the peak is rounder and the reflected wave later and more blended, as in the digital volume pulses of the simulated pulse wave database of Charlton et al. [9].

Rather than a separate model we reuse the Windkessel of the ABP with a third shape row for the finger, with a lower impedance, a wider ejection and a later dicrotic wave:

| site | $c_s$ | $b_s$ | $a_s$ | $c_d$ | $b_d$ | $a_d$ | $z_c$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| finger | 0.30 | 0.14 | 1 | 1.40 | 0.14 | 0.30 | 0.02 |

The foot is delayed by a further radial-to-finger transit $\Delta_{rf} \sim \mathcal{U}(0.03, 0.08)$ s, so the foot of beat $k$ is at $r_k + p + \Delta_{ab} + \Delta_{br} + \Delta_{rf}$, between 0.18 and 0.40 s after the R wave, and the notch is shifted by the same amount.
These are initial values, tuned by eye against the radial shape at 40, 70 and 140 bpm.

The pleth has no absolute units, as a monitor rescales it to fit the screen, so the shape is rescaled over the nominal window to $[0, 1]$ and set to a drawn height about a zero mean:

$$\mathrm{PPG}_0(t) = A\,\bigl(s(t) - \bar s\bigr), \qquad A \sim \mathcal{U}(0.5, 1.5).$$

Respiration modulates the PPG in three ways, baseline wander, amplitude modulation and frequency modulation [10].
The frequency modulation is already in the trace through the RSA of the R-wave walk.
The other two are added reading the same $x(t)$ as every other trace:

$$\mathrm{PPG}(t) = \mathrm{PPG}_0(t)\,\bigl(1 - m\, x(t)\bigr) - w\, x(t) + \sigma\, \epsilon(t),$$

with $m \sim \mathcal{U}(0.05, 0.30)$ so the pulse height falls at end-inspiration as the stroke volume does, $w \sim \mathcal{U}(0, 0.3)$ in the units of $A$ so the baseline falls at end-inspiration as venous return empties the bed, and $\sigma \sim \mathcal{U}(0.005, 0.03)$.

The model has no venous component in the pulse, no motion artefact, and no DC level or perfusion index, since only the AC pulse is drawn.
The skin pulse of the neck that the video shows is built from this same finger shape and these draws of $m$ and $w$, but with its foot at the carotid rather than the finger, at unit height and without noise; it is described in [frame rendering](frame_rendering.md).

## References

[1] P. E. McSharry, G. D. Clifford, L. Tarassenko and L. A. Smith, "A dynamical model for generating synthetic electrocardiogram signals," *IEEE Transactions on Biomedical Engineering*, vol. 50, no. 3, pp. 289–294, 2003. doi:10.1109/TBME.2003.808805. Reference implementation: [ECGSYN on PhysioNet](https://physionet.org/content/ecgsyn/1.0.0/).

[2] A. M. Weissler, W. S. Harris and C. D. Schoenfeld, "Systolic time intervals in heart failure in man," *Circulation*, vol. 37, no. 2, pp. 149–159, 1968. doi:10.1161/01.CIR.37.2.149.

[3] N. Westerhof, J.-W. Lankhaar and B. E. Westerhof, "The arterial Windkessel," *Medical & Biological Engineering & Computing*, vol. 47, no. 2, pp. 131–141, 2009. doi:10.1007/s11517-008-0359-2.

[4] ISO 81060-3:2022, *Non-invasive sphygmomanometers — Part 3: Clinical investigation of continuous automated measurement type*, clause 4.3.3, Blood pressure distribution.

[5] J. B. Mark, "Central venous pressure monitoring: clinical insights beyond the numbers," *Journal of Cardiothoracic and Vascular Anesthesia*, vol. 5, no. 2, pp. 163–173, 1991. doi:10.1016/1053-0770(91)90333-O.

[6] T. Korakianitis and Y. Shi, "A concentrated parameter model for the human cardiovascular system including heart valve dynamics and atrioventricular interaction," *Medical Engineering & Physics*, vol. 28, no. 7, pp. 613–628, 2006. doi:10.1016/j.medengphy.2005.10.004.

[7] A. Pironet, P. C. Dauby, S. Paeme, S. Kosta, J. G. Chase and T. Desaive, "Simulation of left atrial function using a multi-scale model of the cardiovascular system," *PLoS ONE*, vol. 8, no. 6, e65146, 2013. doi:10.1371/journal.pone.0065146.

[8] J. Allen, "Photoplethysmography and its application in clinical physiological measurement," *Physiological Measurement*, vol. 28, no. 3, pp. R1–R39, 2007. doi:10.1088/0967-3334/28/3/R01.

[9] P. H. Charlton, J. Mariscal Harana, S. Vennin, Y. Li, P. Chowienczyk and J. Alastruey, "Modeling arterial pulse waves in healthy aging: a database for in silico evaluation of hemodynamics and pulse wave indexes," *American Journal of Physiology-Heart and Circulatory Physiology*, vol. 317, no. 5, pp. H1062–H1085, 2019. doi:10.1152/ajpheart.00218.2019.

[10] P. H. Charlton, T. Bonnici, L. Tarassenko, D. A. Clifton, R. Beale and P. J. Watkinson, "An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram," *Physiological Measurement*, vol. 37, no. 4, pp. 610–626, 2016. doi:10.1088/0967-3334/37/4/610.
