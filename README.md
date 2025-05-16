# Training Data 

## Injections and Light Curve Data

### Injections Files

We provide .dat files corresponding to the O4 and O5 observing runs. These files contain information on:

Binary component masses and spins for BNS and NSBH systems
Localization parameters: right ascension, declination, and inclination angle
Luminosity distance
O4 refers to the current LIGO-Virgo-KAGRA observing run, while O5 represents projected data from future observing campaigns.

Light Curve Files
Each file contains simulated light curves with:

Magnitudes across various filters: u, g, r, i, y, z, J, H, and K.
A simulation ID matching the IDs in the injection files
This ensures traceability between each light curve and its corresponding binary system parameters.

## Generating Light Curves Using NMMA

To generate kilonova light curves from injection files, use the NMMA repository (https://github.com/np699/nmma). Follow these steps:

### 1. Set Up the NMMA Environment
Follow the instructions in the official NMMA documentation:
👉 NMMA Setup Guide (https://nuclear-multimessenger-astronomy.github.io/nmma/)

Make sure to install the environment and dependencies correctly.

### 2. Create a JSON Injection File
Convert a .dat injection file into a JSON format required by NMMA using the following command:

```bash
nmma-create-injection \
  --prior-file ./Bu2019lm.prior \
  --injection-file ./bns_O4_injections.dat \
  --eos-file ./example_files/eos/ALF2.dat \
  --binary-type BNS \
  --extension json \
  -f ./outdir/injection_Bu2019lm \
  --generation-seed 42 \
  --aligned-spin \
  --eject
```

#### Notes:

Use bns_O4_injections.dat for BNS and nsbh_O4_injections.dat for NSBH.
The --eject flag enables kilonova ejecta parameter conversion.
The Bu2019lm.prior file must contain ratio_zeta, ratio_epsilon, and alpha when --eject is used.

### 3. Generate Light Curves
Once you have the JSON injection file, run:

```bash
light_curve_analysis \
  --model {model} \
  --interpolation_type sklearn_gp \
  --svd-path {svd_path} \
  --outdir {outdir_new} \
  --label injection_{model} \
  --prior {prior_file} \
  --tmin 0.1 --tmax 14 --dt 0.2 \
  --error-budget 1 \
  --nlive 512 \
  --Ebv-max 0 \
  --injection {inject_file} \
  --injection-num {ii} \
  --injection-outfile {inject_outdir}/lc.csv \
  --generation-seed {seed} \
  --filters ztfg,ztfr,ztfi,ps1__z,ps1__y,sdssu,2massh,2massj,2massks \
  --ylim '30,14' \
  --plot \
  --train-stats
```

Replace placeholders in {} with your specific paths or values.

## All-Sky Event Parameters and EM-Bright Classification
The allsky.dat files include additional parameters for each event, such as: Signal-to-noise ratio (SNR), 90% credible area (area(90)), and other localization metrics.
To determine the electromagnetic-bright (EM-Bright) classification of an event, use the em-bright classification tool provided by the LIGO Collaboration (https://git.ligo.org/emfollow/em-properties/em-bright). This tool outputs key physical properties such as:

- HasNS: Indicates the presence of at least one neutron star
- HasRemnant: Suggests the existence of a remnant after merger
- HasMassGap: Identifies systems with component masses in the 3–5 solar mass range

## Links to Data and Model 
### ZTF
Link to the model without mass_ejecta: https://drexel0-my.sharepoint.com/:u:/g/personal/np699_drexel_edu/Ebur3uVzvMRAkxKq6ySJMukBLqpZVrkKv9CdJ1PxzAowOA?e=s2oyBp
Link to the model with mass_ejecta: https://drexel0-my.sharepoint.com/:u:/g/personal/np699_drexel_edu/EddRLVicdFJLjz-PqIlejAABpK2FarM3wXdSCyrFcWxeWw?e=xKIpc9
Link to O4 run dataset: https://drexel0-my.sharepoint.com/:f:/g/personal/np699_drexel_edu/Eiiu84uoD5NAheUAcCORKyYBh29DRfSRAz6_4DKebExHpQ?e=g3Erkj

### Rubin
Link to O5 run dataset: https://drexel0-my.sharepoint.com/:f:/g/personal/np699_drexel_edu/ElbV56VDxwtPtLro1i4UEPsBAym0O8PrpZRUXlP1GQBNKA?e=42WJrX
Link to the model without mass_ejecta: https://drexel0-my.sharepoint.com/:u:/g/personal/np699_drexel_edu/EZDc89uG-TNOn7wFU6UAMz0BwTUSwR0NZJ6kjT6Lv3FKLw?e=ob2c6o
Link to the model with mass_ejecta: https://drexel0-my.sharepoint.com/:u:/g/personal/np699_drexel_edu/EeKhlyaYz6RFrNwEywBw9d8BDGxDIb6XrfQF334s2Juvtg?e=3C3BXC
