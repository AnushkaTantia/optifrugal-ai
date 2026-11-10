# Data provenance (OptiFrugal AI)

## Sourced from public material (checked Nov 2026)
| Quantity | Value used | Source |
|---|---|---|
| CR-39: n 1.498, Abbe 58, density 1.32 | as listed | All About Vision lens guide; Laramy-K lens materials chart |
| Trivex: n 1.53, Abbe 43-45 (44 used), density 1.11 | as listed | Eyecare Business (PPG Trivex introduction); Laramy-K; All About Vision |
| Polycarbonate: n 1.59, Abbe 28-30 (30 used), density 1.21 | as listed | Mitsui Chemicals MR-series comparison table; All About Vision; Younger Optics (specific gravity 1.21) |
| Mid-index 1.56: Abbe 34-36 (35 used) | as listed | Mitsui Chemicals MR-series comparison table |
| MR-8: n 1.60, Abbe 41, density 1.30 | as listed | Mitsui Chemicals (Abbe, heat distortion); Younger Optics TechSpec 2020 (density 1.30) |
| MR-7: n 1.67, Abbe 31, density 1.35 | as listed | Mitsui Chemicals; Fuz/Mitsui MR guide (1.35-1.36) |
| MR-174: n 1.74, Abbe 32, density 1.47 | as listed | ZEISS ClearView lens specification (vd 32, 1.47 g/cm3); Mitsui Chemicals |
| Impact strength (in-lbf): CR-39 75, 1.60 high-index 89, 1.67 high-index 97, Trivex and polycarbonate above 340 (test limit) | scaled so CR-39 = 0.60 | Hoya Phoenix white paper (PPG data). The 340 figure is a lower bound, and the 1.60/1.67 values are for generic high-index lenses |
| Relative price ladder 1.60 : 1.67 : 1.74 = 1 : 1.45 : 2.14 | used to set cost per gram | Lensology single-vision anti-glare price list (UK) |
| Indian retail ratio 1.56 : 1.60 (MR-8) : 1.67 = 1 : 2.6 : 2.7 | used to anchor the MR-8 cost | Medplus Tulip lens price list (India) |
| Edge thickness minimum 1.0 mm, centre thickness minimum 1.0 mm | practical limits | Laramy-K (Trivex passes the FDA drop-ball test at 1.0 mm centre thickness) |

## Still assumptions (flag these in any presentation)
- Cost per gram for CR-39, 1.56, polycarbonate, Trivex; all processing and coating costs; the 1.3 yield factor
- CO2e factors, recyclability and hardness scores
- Impact strength for 1.56 and MR-174; the impact-vs-thickness exponent (0.8)
- The optical-quality formula (Tscherning-like best form, chromatic aberration weighting)
- Retail price ratios are used as a proxy for manufacturing cost
