# Chapter 10 Time Series Analysis

## Slide 1

Chapter 10:
Time Series Analysis
Prachya Boonkwan (Arm)
School of ICT, SIIT,
Thammasat University
prachya@siit.tu.ac.th, kaamanita@gmail.com 

![Image 1](images/slide_1_img_1.jpeg)

---

## Slide 2

https://tinyurl.com/5b6ch5px

![Image 1](images/slide_2_img_1.png)

---

## Slide 3

Who? Me?
- Nickname: Arm (P’/N’ Arm, etc.)
- Born: Aug 1981
- Work
- Researcher at NECTEC 2005-2024
- Lecturer at SIIT, Thammasat University 2025-now
- Education
- B.Eng & M.Eng, CPE Kasetsart University
- Obtained Ministry of Science Scholarship in early 2008
- Did a PhD in Informatics (AI & Computational Linguistics) at 
University of Edinburgh, UK from 2008 to 2013 (4.5 years)

![Image 1](images/slide_3_img_1.jpeg)

---

## Slide 4

Outline
- Introduction
- Time-domain methods
- Autoregressive integrated moving average (ARIMA)
- Convolutional analysis and CNNs
- Frequency-domain methods
- Spectral density analysis
- Wavelet analysis
- Transformer for time series
- Conclusion

---

## Slide 5

1. Introduction

---

## Slide 6

Time Series
- A sequence of data points read at equally spaced 
points of time
- E.g. daily stock price, monthly rice price, heights of ocean 
tides, audio signals, counts of celestial meteriorites, and 
activity of tectonic plates
- Time series forecasting
- Predicting future values based on previously observed 
values
- Stochastic process: observations close together in time  
are more closely related than those further apart

![Image 1](images/slide_6_img_1.png)

---

## Slide 7

Stationary vs. Non-Stationary
- Stationary time series is one whose statistical 
properties do not change over time
- Constant mean: average value remains the same 
throughout the time series
- Constant variance: spread of data points do not change
- Constant autocovariance: relationship between past and 
present values depends on the lag, not the time
- Non-stationary time series is one whose any of 
these properties change over time
Credit: https://medium.com/codex/what-is-stationarity-in-time-series-how-it-can-be-detected-7e5dfa7b5f6b

![Image 1](images/slide_7_img_1.jpeg)

---

## Slide 8

Stationary vs. Non-Stationary
- Non-stationary time series is one whose mean, variance 
(spread), or autocovariance (lag) change over time
Credit: https://medium.com/codex/what-is-stationarity-in-time-series-how-it-can-be-detected-7e5dfa7b5f6b

![Image 1](images/slide_8_img_1.jpeg)

---

## Slide 9

Patterns in Time Series
- Trend: general direction over time
- Seasonality: repetitive patterns that occur at 
regular predictable intervals
- Holiday effects: irregular patterns caused by 
special calendar events
- Cycle: long-term repetitive patterns that occur 
at irregular intervals
time (t)
holiday effect
time (t)
trend
Cycle
1. Slow increase
2. Catastrophe
3. Rapid decline
season

---

## Slide 10

Assumptions about Time Series
- Time domain
- A1: Stochastic process (observations closer in 
time are more closely related)
- A2: Combination of temporal structures
- Frequency domain
- A3: Combination of continuous waves
- A4: Combination of wavelets (i.e. wavelike pieces)
- Sequence-to-sequence prediction
- A5: Transformer-based models

---

## Slide 11

2. Time-Domain Methods

---

## Slide 12

2.1 ARIMA Model

---

## Slide 13

ARIMA Model
- Auto-Regressive Integrated Moving Average
- Assumption: The dataset is seasonal and the difference 
between seasons can be predicted by linear regression
- How: Predict a future value across the season by linear 
regression of previous cross-seasonal differences and 
adjust the error by linear regression of previous errors
- Three parameters of ARIMA(p,d,q)
- Season duration: d timesteps
- No. cross-seasonal differences: p timesteps
- No. previous errors: q timesteps
time (t)
season
time (t)

---

## Slide 14

ARIMA(p,d,q)
- Step 1: Integrate the season of length d 
for every point xt in the time series
- Example: 
Season length d = 3
time (t)
season
time (t)
x→
t = xt →xt↑d
t
xt
1
10
2
15
3
20
4
25
5
28
6
38

---

## Slide 15

ARIMA(p,d,q)
- Step 1: Integrate the season of length d 
for every point xt in the time series
- Example: 
Season length d = 3
time (t)
season
time (t)
x→
t = xt →xt↑d
t
xt
xt-d
x't
1
10
—
—
2
15
—
—
3
20
—
—
4
25
10
15
5
28
15
13
6
38
20
18

---

## Slide 16

ARIMA(p,d,q)
- Step 1: Integrate the season of length d 
- Step 2: Predict the current difference with a linear 
regression of p previous differences
The term ‘autoregressive’ means taking the recent 
outputs as inputs for the next computation
time (t)
x→
t = xt →xt↑d
x→
t =
! p
"
k=1
ωkx→
t↑k
#
+ et
time (t)
predicted
difference
error
predicted
difference
error
= xt – (xt–d + diff)

---

## Slide 17

ARIMA(p,d,q)
- Step 1: Integrate the season of length d 
- Step 2: Predict the current difference with a linear 
regression of p previous differences
- Step 3: Adjust the prediction error with a linear 
regression of q previous errors
x→
t = xt →xt↑d
x→
t =
! p
"
k=1
ωkx→
t↑k
#
+ et
sha1_base64="Std
Unia5g9+f4/ZOls
EA27pu/0=">ACU
nicbVJbSxtBFJ7ES2
3qJbWPvhwMtoIYdk
WtL4LUFx8VjArZ7T
I7OZud7OylM2dLw5
LfKJS+EN86YM6iUG
8HRj4+C7MzDcTFko
acpybWn1mdm7+w8L
HxqfFpeWV5ufVC5O
XWmBH5CrXVyE3qGS
GHZKk8KrQyNQ4WY
HI/1y9+ojcyzcxoW
6Ke8n8lICk6WCpry
z7eA4Cscgqcwoi54
pkyDKjl0Rz8L8IpYB
glYT0XbyQg8Lfsx+
bAFaFNbLzODceYXe
BQj8WBgLTY0eAoFz
ZbTdiYDb4E7BS02nd
Og+dfr5aJMSOhuD
Fd1ynIr7gmKRSOGl
5psOAi4X3sWpjxFI
1fTSoZwYZlehDl2q6
MYMI+T1Q8NWaYhta
ZcorNa21Mvqd1S4o
O/EpmRUmYiceNolI
B5TDuF3pSoyA1tIA
Le1ZQcRc0H2FRq2
BPf1ld+Ci52u9/e
O9tHf2Y1rHA1tg6
2Qu+86O2Ak7ZR0m
2DW7ZXfsvav9r9uf
8mjtV6bZr6wF1Nf
Af3rJQ</latexit
>
x→
t =
! p
"
k=1
ωkx→
t↑k
#
+ et +


q
"
j=1
εjet↑j


time (t)
predicted
difference
error
time (t)
predicted
difference
adj. error

---

## Slide 18

Training Algorithm of ARIMA(p,d,q)
- Suppose each data point xt is in the training set
- Compute the differences x't for each data point
- Estimate parameters φk with MSE of all et
- Estimate parameters θk with MSE of all et with fixed φk
x→
t = xt →xt↑d
et = x→
t →
! p
"
k=1
ωkx→
t↑k
#
et = x→
t →
! p
"
k=1
ωkx→
t↑k
#
→


q
"
j=1
εjet↑j


MSE = 1
N
N
!
t=1
e2
t

---

## Slide 19

Prediction of ARIMA(p,d,q)
- Suppose we want to predict future values xt+1 to xt+N
- We compute the differences and errors of timesteps t+1 to t+N 
- We compute the future values from the differences
xt = x→
t + xt↑d
2bFAyH4ejz4g=">ACnHicfVFba9RAFJ7EW10vXfWpCHJw8QLSJRFvL6VFfaiIUMHdFj
YxTGZPNrOZXJw5KV1CfpX/xDf/jZNtHnSrHhj4+L7vXOacuFLSkOf9dNxLl69cvbZ1fXDj
5q3b28M7d6emrLXAiShVqU9iblDJAickSeFJpZHnscLjOHvX6cenqI0siy+0qjDM+aKQiR
ScLBUNv589iQgewx4Eps6jJtvz268VBFUqowys2NBu1sKzXl528jcIKEXi0RKw05ctBMEA
+zrgrsQKExo9v+qgZaLlMJN9z+b9PZoOPLG3jrgIvB7MGJ9HEXDH8G8FHWOBQnFjZn5Xk
VhwzVJobAdBLXBiouML3BmYcFzNGzXm4Ljywzh6TU9hUEa/b3jIbnxqzy2DpzTqnZ1Dryb
9qspuRN2MiqgkLcd4oqRVQCd2lYC41ClIrC7jQ0s4KIuWaC7L3HNgl+Jtfvgimz8f+q/H
Lzy9GB2/7dWyx+whe8p89podsEN2xCZMODvOvnPofHAfuO/dj+6nc6vr9Dn32B/hTn8Bu
nrIKQ=</latexit>
x→
t =
p
!
k=1
ωkx→
t↑k +
q
!
j=1
εjet↑j
et = x→
t →
" p
!
k=1
ωkx→
t↑k
#
→


q
!
j=1
εjet↑j



---

## Slide 20

Incomplete Time Series
- Interpolation techniques
- Newton’s interpolation: each ai is computed by divided differences
- Cubic spline interpolation: cubic curve
- Chebyshev’s interpolation: sinusoidal seasonality
- Radial basis function interpolation: spread around the means xi
N(x) = a0 + a1(x →x0) + a2(x →x0)(x →x1) + . . . + an
n→1
!
k=0
(x →xk)
Si(x) = ai + bi(x →xi) + ci(x →xi)2 + di(x →xi)3
xi = a + b
2
+ b →a
2
cos
!2i + 1
2n
ω
"
R(x) =
n
!
i=1
ωi · ε (|x →xi|)

---

## Slide 21

Evaluation of Prediction Models
- Mean absolute error (MAE): average absolute 
difference between prediction and gold standard
- Root mean squared error (RMSE): square root of 
mean squared difference between prediction and 
gold standard
- Mean absolute percentage error (MAPE): average 
percentage difference between prediction and gold 
standard (w.r.t. gold standard)
- Forecast bias: average bias (prediction – gold 
standard)
MAE = 1
N
N
!
k=1
|yk →ˆyk|
QEi9EKUtAaRsWHkdb2LF3t3ab5FW1v4xLvwHbr31wgGEuHKtE3JoSNZGs28p+eZOJfCoO/8mZmX7x8NTf/uvbm7bv3C/UPi8cmKzTjHZbJTJ/G1HApUt5BgZKf5pTFUt+Eo/2x/7JOdGZOkRljnvKTpIRSIYRSdF9aNQURxqZb8f/jio4BPsQGh+arQ
QJpoyG1S2VTmpUJEd7QTVWQtCyRNchTIawRcIhxRtWTkeajEY4uezNaiesNv+hPAcxJMSYNM0Y7qV2E/Y4XiKTJjekGfo49SzUKJnlVCwvDc8pGdMC7jqZUcdOzk/QVfHRKH5JMu5ciTNS/NyxVxpQqdpPjrOapNxb/53ULTLZ7VqR5gTxlj4eSQgJmMK
4S+kJzhrJ0hDIt3F+BDalrDV3hNVdC8DTyc3K81gw2mxvf1hu7e9M65skyWSGrJCBbZJd8JW3SIYxckN/khtx6l961d+fdP47OeNOdJfIPvIc/aoCtjA=</latexit>
RMSE =
!
"
"
# 1
N
N
$
k=1
(yk →ˆyk)2
I18XgJhR6qXEg/gDLFav1yl60K4ndp4DY6J/l0r+QW6+9NAQes0ta1uH1u7AwjDzHm9nwkxwDa7w9l48vTZ82tF7Xtl69e79R39/o6zRVlPZqKVA1DopngCesB8GmWJEhoINwvh87g+umNI8TS6hyNhYkmnCI04JWCmo931JYKak+fqp
+7nE73EL+5Ei1Hil6ZTY17kMTNzym8d7AsWwXl4yKI8UfszwiYogzi0ljBLig+ncF1UG+4TXcBvE68ijRQhW5Qv/UnKc0lS4AKovXIczMYG6KAU8HKmp9rlhEakykbWZoQyfTYLPKXeN8qExylyr4E8EL9e8MQqXUhQzs5T6tXvbn4P2+UQ
3Q6NjzJcmAJXR6KcoEhxfMy8YQrRkEUlhCquP0rpjNi6wFbec2W4K1GXif9g6Z3Dy6OGy0z6o6tBb9A59QB46QW30BXVRD1F0g36i3+jO+e78cu6dP8vRDafaeYP+gfPwCOHlr5A=</latexit>
MAPE = 1
N
N
!
k=1
""""
yk →ˆyk
yk
""""
bias = 1
N
N
!
k=1
(yk →ˆyk)

---

## Slide 22

2.2 Convolutional Analysis and CNNs

---

## Slide 23

Convolution ∗
- Combining two functions to produce a third 
one that represents how one amplifies the 
other in each step of their overlapping
- Continuous convolution
- Discrete convolution
- where f is an input signal and w is a kernel (i.e. 
pattern filter)
Kernel w
(flipped)
Input f
(as is)
t=0
t=1
t=2
t=3
[f →w](t) =
! →
↑→
f(ω) · w(t ↑ω) dω
[f →w](t) =
→
!
k=↑→
f[k] · w[t ↑k]
Two waves are colliding
This is our kernel w

---

## Slide 24

Cross-Correlation ⊙
- The intuitive counterpart of convolution, 
where the kernel function is not flipped
- Continuous cross-correlation
- Discrete cross-correlation
- We will say ‘convolution’ to refer to cross-
correlation for ease of understanding
This is our kernel w
="JFLzgJMEF0iSQulDj/moU1vDX+k=">ACOXi
cbZDNSiQxFIVTjo7aOmPuHRzsVHaEZsqmb+NIO
PGZQu2Cl1lk0qlNJhKiuTWSFP0a83Gt3AnuJmFI
m59AVNtLfy7EPJxzr0k98S5FBZ9/8qb+DA59XF6
ZrYxN/p80Lzy9cDqwvDeI9pqc1RTC2XQvEeCpT
8KDecZrHkh/HZTuUf/uXGCq32cZjzKMnSqSCUX
TSoNntpxDqRCOcR21cg1XYglAoHJQb7kpxODoua
4C0HSIt1iBkVX953kZYh7E0Ov4GISQVD5otv+OP
C95CUEOL1NUdNC/DRLMi4wqZpNb2Az/HqKQGBZN
81AgLy3PKzugJ7ztUNOM2Ksebj2DFKQmk2rijEM
bq84mSZtYOs9h1ZhRP7WuvEt/z+gWmv6NSqLxAr
tjTQ2khATVUMUIiDGcohw4oM8L9FdgpNZShC7vh
Qgher/wWDjY7wc/Oj73vre0/dRwzZIkskzYJyC+
yTXZJl/QI/INbkht96F9+78+6fWie8emaRvC
jv4RGa7KsH</latexit>
[f →w](t) =
! →
↑→
f(ω) · w(t + ω)↓dω
[f →w](t) =
→
!
k=↑→
f[k] · w[t + k]↓
* is the complex conjugate
e.g. (3 + 4î)* = 3 – 4î
Similarity of two waves
Kernel w
(as is)
Input f
(as is)
t=0
t=1
t=2
t=3

---

## Slide 25

Convolutional Filtering
Find
(kernel)

![Image 1](images/slide_25_img_1.png)

![Image 2](images/slide_25_img_2.png)

![Image 3](images/slide_25_img_3.png)

![Image 4](images/slide_25_img_4.png)

![Image 5](images/slide_25_img_5.png)

![Image 6](images/slide_25_img_6.png)

![Image 7](images/slide_25_img_7.png)

![Image 8](images/slide_25_img_8.png)

![Image 9](images/slide_25_img_9.png)

![Image 10](images/slide_25_img_10.png)

![Image 11](images/slide_25_img_11.png)

![Image 12](images/slide_25_img_12.png)

![Image 13](images/slide_25_img_13.png)

![Image 14](images/slide_25_img_14.png)

![Image 15](images/slide_25_img_15.png)

![Image 16](images/slide_25_img_16.png)

![Image 17](images/slide_25_img_17.png)

![Image 18](images/slide_25_img_18.png)

![Image 19](images/slide_25_img_19.png)

![Image 20](images/slide_25_img_20.png)

![Image 21](images/slide_25_img_21.png)

![Image 22](images/slide_25_img_22.png)

![Image 23](images/slide_25_img_23.png)

![Image 24](images/slide_25_img_24.png)

![Image 25](images/slide_25_img_25.png)

![Image 26](images/slide_25_img_26.png)

![Image 27](images/slide_25_img_27.png)

![Image 28](images/slide_25_img_28.png)

![Image 29](images/slide_25_img_29.png)

![Image 30](images/slide_25_img_30.png)

![Image 31](images/slide_25_img_31.png)

![Image 32](images/slide_25_img_32.png)

![Image 33](images/slide_25_img_33.png)

![Image 34](images/slide_25_img_34.png)

![Image 35](images/slide_25_img_35.png)

![Image 36](images/slide_25_img_36.png)

![Image 37](images/slide_25_img_37.png)

![Image 38](images/slide_25_img_38.png)

![Image 39](images/slide_25_img_39.png)

![Image 40](images/slide_25_img_40.png)

![Image 41](images/slide_25_img_41.png)

![Image 42](images/slide_25_img_42.png)

![Image 43](images/slide_25_img_43.png)

![Image 44](images/slide_25_img_44.png)

![Image 45](images/slide_25_img_45.png)

![Image 46](images/slide_25_img_46.png)

![Image 47](images/slide_25_img_47.png)

![Image 48](images/slide_25_img_48.png)

![Image 49](images/slide_25_img_49.png)

![Image 50](images/slide_25_img_50.png)

![Image 51](images/slide_25_img_51.png)

![Image 52](images/slide_25_img_52.png)

![Image 53](images/slide_25_img_53.png)

![Image 54](images/slide_25_img_54.png)

![Image 55](images/slide_25_img_55.png)

![Image 56](images/slide_25_img_56.png)

![Image 57](images/slide_25_img_57.png)

![Image 58](images/slide_25_img_58.png)

![Image 59](images/slide_25_img_59.png)

![Image 60](images/slide_25_img_60.png)

![Image 61](images/slide_25_img_61.png)

![Image 62](images/slide_25_img_62.png)

![Image 63](images/slide_25_img_63.png)

![Image 64](images/slide_25_img_64.png)

![Image 65](images/slide_25_img_65.png)

![Image 66](images/slide_25_img_66.png)

![Image 67](images/slide_25_img_67.png)

![Image 68](images/slide_25_img_68.png)

![Image 69](images/slide_25_img_69.png)

![Image 70](images/slide_25_img_70.png)

![Image 71](images/slide_25_img_71.png)

![Image 72](images/slide_25_img_72.png)

![Image 73](images/slide_25_img_73.png)

![Image 74](images/slide_25_img_74.png)

![Image 75](images/slide_25_img_75.png)

![Image 76](images/slide_25_img_76.png)

![Image 77](images/slide_25_img_77.png)

![Image 78](images/slide_25_img_78.png)

![Image 79](images/slide_25_img_79.png)

![Image 80](images/slide_25_img_80.png)

![Image 81](images/slide_25_img_81.png)

![Image 82](images/slide_25_img_82.png)

![Image 83](images/slide_25_img_83.png)

![Image 84](images/slide_25_img_84.png)

![Image 85](images/slide_25_img_85.png)

![Image 86](images/slide_25_img_86.png)

![Image 87](images/slide_25_img_87.png)

![Image 88](images/slide_25_img_88.png)

![Image 89](images/slide_25_img_89.png)

![Image 90](images/slide_25_img_90.png)

![Image 91](images/slide_25_img_91.png)

![Image 92](images/slide_25_img_92.png)

![Image 93](images/slide_25_img_93.png)

![Image 94](images/slide_25_img_94.png)

![Image 95](images/slide_25_img_95.png)

![Image 96](images/slide_25_img_96.png)

![Image 97](images/slide_25_img_97.png)

![Image 98](images/slide_25_img_98.png)

![Image 99](images/slide_25_img_99.png)

![Image 100](images/slide_25_img_100.png)

![Image 101](images/slide_25_img_101.png)

![Image 102](images/slide_25_img_102.png)

![Image 103](images/slide_25_img_103.png)

![Image 104](images/slide_25_img_104.png)

![Image 105](images/slide_25_img_105.png)

![Image 106](images/slide_25_img_106.png)

![Image 107](images/slide_25_img_107.png)

![Image 108](images/slide_25_img_108.png)

![Image 109](images/slide_25_img_109.png)

![Image 110](images/slide_25_img_110.png)

![Image 111](images/slide_25_img_111.png)

![Image 112](images/slide_25_img_112.png)

![Image 113](images/slide_25_img_113.png)

![Image 114](images/slide_25_img_114.png)

![Image 115](images/slide_25_img_115.png)

![Image 116](images/slide_25_img_116.png)

![Image 117](images/slide_25_img_117.png)

![Image 118](images/slide_25_img_118.png)

![Image 119](images/slide_25_img_119.png)

![Image 120](images/slide_25_img_120.png)

![Image 121](images/slide_25_img_121.png)

![Image 122](images/slide_25_img_122.png)

![Image 123](images/slide_25_img_123.png)

![Image 124](images/slide_25_img_124.png)

![Image 125](images/slide_25_img_125.png)

![Image 126](images/slide_25_img_126.png)

![Image 127](images/slide_25_img_127.png)

![Image 128](images/slide_25_img_128.png)

![Image 129](images/slide_25_img_129.png)

![Image 130](images/slide_25_img_130.png)

![Image 131](images/slide_25_img_131.png)

![Image 132](images/slide_25_img_132.png)

![Image 133](images/slide_25_img_133.png)

![Image 134](images/slide_25_img_134.png)

![Image 135](images/slide_25_img_135.png)

![Image 136](images/slide_25_img_136.png)

![Image 137](images/slide_25_img_137.png)

![Image 138](images/slide_25_img_138.png)

![Image 139](images/slide_25_img_139.png)

![Image 140](images/slide_25_img_140.png)

![Image 141](images/slide_25_img_141.png)

![Image 142](images/slide_25_img_142.png)

![Image 143](images/slide_25_img_143.png)

![Image 144](images/slide_25_img_144.png)

![Image 145](images/slide_25_img_145.png)

![Image 146](images/slide_25_img_146.png)

![Image 147](images/slide_25_img_147.png)

![Image 148](images/slide_25_img_148.png)

![Image 149](images/slide_25_img_149.png)

![Image 150](images/slide_25_img_150.png)

![Image 151](images/slide_25_img_151.png)

![Image 152](images/slide_25_img_152.png)

![Image 153](images/slide_25_img_153.png)

![Image 154](images/slide_25_img_154.png)

![Image 155](images/slide_25_img_155.png)

![Image 156](images/slide_25_img_156.png)

![Image 157](images/slide_25_img_157.png)

![Image 158](images/slide_25_img_158.png)

![Image 159](images/slide_25_img_159.png)

![Image 160](images/slide_25_img_160.png)

![Image 161](images/slide_25_img_161.png)

![Image 162](images/slide_25_img_162.png)

![Image 163](images/slide_25_img_163.png)

![Image 164](images/slide_25_img_164.png)

![Image 165](images/slide_25_img_165.png)

![Image 166](images/slide_25_img_166.png)

![Image 167](images/slide_25_img_167.png)

![Image 168](images/slide_25_img_168.png)

![Image 169](images/slide_25_img_169.png)

![Image 170](images/slide_25_img_170.png)

![Image 171](images/slide_25_img_171.png)

![Image 172](images/slide_25_img_172.png)

![Image 173](images/slide_25_img_173.png)

![Image 174](images/slide_25_img_174.png)

![Image 175](images/slide_25_img_175.png)

![Image 176](images/slide_25_img_176.png)

![Image 177](images/slide_25_img_177.png)

![Image 178](images/slide_25_img_178.png)

![Image 179](images/slide_25_img_179.png)

![Image 180](images/slide_25_img_180.png)

![Image 181](images/slide_25_img_181.png)

![Image 182](images/slide_25_img_182.png)

![Image 183](images/slide_25_img_183.png)

![Image 184](images/slide_25_img_184.png)

![Image 185](images/slide_25_img_185.png)

![Image 186](images/slide_25_img_186.png)

![Image 187](images/slide_25_img_187.png)

![Image 188](images/slide_25_img_188.png)

![Image 189](images/slide_25_img_189.png)

![Image 190](images/slide_25_img_190.png)

![Image 191](images/slide_25_img_191.png)

![Image 192](images/slide_25_img_192.png)

![Image 193](images/slide_25_img_193.png)

![Image 194](images/slide_25_img_194.png)

![Image 195](images/slide_25_img_195.png)

![Image 196](images/slide_25_img_196.png)

![Image 197](images/slide_25_img_197.png)

![Image 198](images/slide_25_img_198.png)

![Image 199](images/slide_25_img_199.png)

![Image 200](images/slide_25_img_200.png)

![Image 201](images/slide_25_img_201.png)

![Image 202](images/slide_25_img_202.png)

![Image 203](images/slide_25_img_203.png)

![Image 204](images/slide_25_img_204.png)

![Image 205](images/slide_25_img_205.png)

![Image 206](images/slide_25_img_206.png)

![Image 207](images/slide_25_img_207.png)

![Image 208](images/slide_25_img_208.png)

![Image 209](images/slide_25_img_209.png)

![Image 210](images/slide_25_img_210.png)

![Image 211](images/slide_25_img_211.png)

![Image 212](images/slide_25_img_212.png)

![Image 213](images/slide_25_img_213.png)

![Image 214](images/slide_25_img_214.png)

![Image 215](images/slide_25_img_215.png)

![Image 216](images/slide_25_img_216.png)

![Image 217](images/slide_25_img_217.png)

![Image 218](images/slide_25_img_218.png)

![Image 219](images/slide_25_img_219.png)

![Image 220](images/slide_25_img_220.png)

![Image 221](images/slide_25_img_221.png)

![Image 222](images/slide_25_img_222.png)

![Image 223](images/slide_25_img_223.png)

![Image 224](images/slide_25_img_224.png)

![Image 225](images/slide_25_img_225.png)

![Image 226](images/slide_25_img_226.png)

![Image 227](images/slide_25_img_227.png)

![Image 228](images/slide_25_img_228.png)

![Image 229](images/slide_25_img_229.png)

![Image 230](images/slide_25_img_230.png)

![Image 231](images/slide_25_img_231.png)

![Image 232](images/slide_25_img_232.png)

![Image 233](images/slide_25_img_233.png)

![Image 234](images/slide_25_img_234.png)

![Image 235](images/slide_25_img_235.png)

![Image 236](images/slide_25_img_236.png)

![Image 237](images/slide_25_img_237.png)

![Image 238](images/slide_25_img_238.png)

![Image 239](images/slide_25_img_239.png)

![Image 240](images/slide_25_img_240.png)

![Image 241](images/slide_25_img_241.png)

![Image 242](images/slide_25_img_242.png)

![Image 243](images/slide_25_img_243.png)

![Image 244](images/slide_25_img_244.png)

![Image 245](images/slide_25_img_245.png)

![Image 246](images/slide_25_img_246.png)

![Image 247](images/slide_25_img_247.png)

![Image 248](images/slide_25_img_248.png)

![Image 249](images/slide_25_img_249.png)

![Image 250](images/slide_25_img_250.png)

![Image 251](images/slide_25_img_251.png)

![Image 252](images/slide_25_img_252.png)

![Image 253](images/slide_25_img_253.png)

![Image 254](images/slide_25_img_254.png)

![Image 255](images/slide_25_img_255.png)

![Image 256](images/slide_25_img_256.png)

![Image 257](images/slide_25_img_257.png)

![Image 258](images/slide_25_img_258.png)

![Image 259](images/slide_25_img_259.png)

![Image 260](images/slide_25_img_260.png)

![Image 261](images/slide_25_img_261.png)

![Image 262](images/slide_25_img_262.png)

![Image 263](images/slide_25_img_263.png)

![Image 264](images/slide_25_img_264.png)

![Image 265](images/slide_25_img_265.png)

![Image 266](images/slide_25_img_266.png)

![Image 267](images/slide_25_img_267.png)

![Image 268](images/slide_25_img_268.png)

![Image 269](images/slide_25_img_269.png)

![Image 270](images/slide_25_img_270.png)

![Image 271](images/slide_25_img_271.png)

![Image 272](images/slide_25_img_272.png)

![Image 273](images/slide_25_img_273.png)

![Image 274](images/slide_25_img_274.png)

![Image 275](images/slide_25_img_275.png)

![Image 276](images/slide_25_img_276.png)

![Image 277](images/slide_25_img_277.png)

![Image 278](images/slide_25_img_278.png)

![Image 279](images/slide_25_img_279.png)

![Image 280](images/slide_25_img_280.png)

![Image 281](images/slide_25_img_281.png)

![Image 282](images/slide_25_img_282.png)

![Image 283](images/slide_25_img_283.png)

![Image 284](images/slide_25_img_284.png)

![Image 285](images/slide_25_img_285.png)

![Image 286](images/slide_25_img_286.png)

![Image 287](images/slide_25_img_287.png)

![Image 288](images/slide_25_img_288.png)

![Image 289](images/slide_25_img_289.png)

![Image 290](images/slide_25_img_290.png)

![Image 291](images/slide_25_img_291.png)

![Image 292](images/slide_25_img_292.png)

![Image 293](images/slide_25_img_293.png)

![Image 294](images/slide_25_img_294.png)

![Image 295](images/slide_25_img_295.png)

![Image 296](images/slide_25_img_296.png)

![Image 297](images/slide_25_img_297.png)

![Image 298](images/slide_25_img_298.png)

![Image 299](images/slide_25_img_299.png)

![Image 300](images/slide_25_img_300.png)

![Image 301](images/slide_25_img_301.png)

![Image 302](images/slide_25_img_302.png)

![Image 303](images/slide_25_img_303.png)

![Image 304](images/slide_25_img_304.png)

![Image 305](images/slide_25_img_305.png)

![Image 306](images/slide_25_img_306.png)

![Image 307](images/slide_25_img_307.png)

![Image 308](images/slide_25_img_308.png)

![Image 309](images/slide_25_img_309.png)

![Image 310](images/slide_25_img_310.png)

![Image 311](images/slide_25_img_311.png)

![Image 312](images/slide_25_img_312.png)

![Image 313](images/slide_25_img_313.png)

![Image 314](images/slide_25_img_314.png)

![Image 315](images/slide_25_img_315.png)

![Image 316](images/slide_25_img_316.png)

![Image 317](images/slide_25_img_317.png)

![Image 318](images/slide_25_img_318.png)

![Image 319](images/slide_25_img_319.png)

![Image 320](images/slide_25_img_320.png)

![Image 321](images/slide_25_img_321.png)

![Image 322](images/slide_25_img_322.png)

![Image 323](images/slide_25_img_323.png)

![Image 324](images/slide_25_img_324.png)

![Image 325](images/slide_25_img_325.png)

![Image 326](images/slide_25_img_326.png)

![Image 327](images/slide_25_img_327.png)

![Image 328](images/slide_25_img_328.png)

![Image 329](images/slide_25_img_329.png)

![Image 330](images/slide_25_img_330.png)

![Image 331](images/slide_25_img_331.png)

![Image 332](images/slide_25_img_332.png)

![Image 333](images/slide_25_img_333.png)

![Image 334](images/slide_25_img_334.png)

![Image 335](images/slide_25_img_335.png)

![Image 336](images/slide_25_img_336.png)

![Image 337](images/slide_25_img_337.png)

![Image 338](images/slide_25_img_338.png)

![Image 339](images/slide_25_img_339.png)

![Image 340](images/slide_25_img_340.png)

![Image 341](images/slide_25_img_341.png)

![Image 342](images/slide_25_img_342.png)

![Image 343](images/slide_25_img_343.png)

![Image 344](images/slide_25_img_344.png)

![Image 345](images/slide_25_img_345.png)

![Image 346](images/slide_25_img_346.png)

![Image 347](images/slide_25_img_347.png)

![Image 348](images/slide_25_img_348.png)

![Image 349](images/slide_25_img_349.png)

![Image 350](images/slide_25_img_350.png)

![Image 351](images/slide_25_img_351.png)

![Image 352](images/slide_25_img_352.png)

![Image 353](images/slide_25_img_353.png)

![Image 354](images/slide_25_img_354.png)

![Image 355](images/slide_25_img_355.png)

![Image 356](images/slide_25_img_356.png)

![Image 357](images/slide_25_img_357.png)

![Image 358](images/slide_25_img_358.png)

![Image 359](images/slide_25_img_359.png)

![Image 360](images/slide_25_img_360.png)

![Image 361](images/slide_25_img_361.png)

![Image 362](images/slide_25_img_362.png)

![Image 363](images/slide_25_img_363.png)

![Image 364](images/slide_25_img_364.png)

![Image 365](images/slide_25_img_365.png)

![Image 366](images/slide_25_img_366.png)

![Image 367](images/slide_25_img_367.png)

![Image 368](images/slide_25_img_368.png)

![Image 369](images/slide_25_img_369.png)

![Image 370](images/slide_25_img_370.png)

![Image 371](images/slide_25_img_371.png)

![Image 372](images/slide_25_img_372.png)

![Image 373](images/slide_25_img_373.png)

![Image 374](images/slide_25_img_374.png)

![Image 375](images/slide_25_img_375.png)

![Image 376](images/slide_25_img_376.png)

![Image 377](images/slide_25_img_377.png)

![Image 378](images/slide_25_img_378.png)

![Image 379](images/slide_25_img_379.png)

![Image 380](images/slide_25_img_380.png)

![Image 381](images/slide_25_img_381.png)

![Image 382](images/slide_25_img_382.png)

![Image 383](images/slide_25_img_383.png)

![Image 384](images/slide_25_img_384.png)

![Image 385](images/slide_25_img_385.png)

![Image 386](images/slide_25_img_386.png)

![Image 387](images/slide_25_img_387.png)

![Image 388](images/slide_25_img_388.png)

![Image 389](images/slide_25_img_389.png)

![Image 390](images/slide_25_img_390.png)

![Image 391](images/slide_25_img_391.png)

![Image 392](images/slide_25_img_392.png)

![Image 393](images/slide_25_img_393.png)

![Image 394](images/slide_25_img_394.png)

![Image 395](images/slide_25_img_395.png)

![Image 396](images/slide_25_img_396.png)

![Image 397](images/slide_25_img_397.png)

![Image 398](images/slide_25_img_398.png)

![Image 399](images/slide_25_img_399.png)

![Image 400](images/slide_25_img_400.png)

![Image 401](images/slide_25_img_401.png)

![Image 402](images/slide_25_img_402.png)

![Image 403](images/slide_25_img_403.png)

![Image 404](images/slide_25_img_404.png)

![Image 405](images/slide_25_img_405.png)

![Image 406](images/slide_25_img_406.png)

![Image 407](images/slide_25_img_407.png)

![Image 408](images/slide_25_img_408.png)

![Image 409](images/slide_25_img_409.png)

![Image 410](images/slide_25_img_410.png)

![Image 411](images/slide_25_img_411.png)

![Image 412](images/slide_25_img_412.png)

![Image 413](images/slide_25_img_413.png)

![Image 414](images/slide_25_img_414.png)

![Image 415](images/slide_25_img_415.png)

![Image 416](images/slide_25_img_416.png)

![Image 417](images/slide_25_img_417.png)

![Image 418](images/slide_25_img_418.png)

![Image 419](images/slide_25_img_419.png)

![Image 420](images/slide_25_img_420.png)

![Image 421](images/slide_25_img_421.png)

![Image 422](images/slide_25_img_422.png)

![Image 423](images/slide_25_img_423.png)

![Image 424](images/slide_25_img_424.png)

![Image 425](images/slide_25_img_425.png)

![Image 426](images/slide_25_img_426.png)

![Image 427](images/slide_25_img_427.png)

![Image 428](images/slide_25_img_428.png)

![Image 429](images/slide_25_img_429.png)

![Image 430](images/slide_25_img_430.png)

![Image 431](images/slide_25_img_431.png)

![Image 432](images/slide_25_img_432.png)

![Image 433](images/slide_25_img_433.png)

![Image 434](images/slide_25_img_434.png)

![Image 435](images/slide_25_img_435.png)

![Image 436](images/slide_25_img_436.png)

![Image 437](images/slide_25_img_437.png)

![Image 438](images/slide_25_img_438.png)

![Image 439](images/slide_25_img_439.png)

![Image 440](images/slide_25_img_440.png)

![Image 441](images/slide_25_img_441.png)

![Image 442](images/slide_25_img_442.png)

![Image 443](images/slide_25_img_443.png)

![Image 444](images/slide_25_img_444.png)

![Image 445](images/slide_25_img_445.png)

![Image 446](images/slide_25_img_446.png)

![Image 447](images/slide_25_img_447.png)

![Image 448](images/slide_25_img_448.png)

![Image 449](images/slide_25_img_449.png)

![Image 450](images/slide_25_img_450.png)

![Image 451](images/slide_25_img_451.png)

![Image 452](images/slide_25_img_452.png)

![Image 453](images/slide_25_img_453.png)

![Image 454](images/slide_25_img_454.png)

![Image 455](images/slide_25_img_455.png)

![Image 456](images/slide_25_img_456.png)

![Image 457](images/slide_25_img_457.png)

![Image 458](images/slide_25_img_458.png)

![Image 459](images/slide_25_img_459.png)

![Image 460](images/slide_25_img_460.png)

![Image 461](images/slide_25_img_461.png)

![Image 462](images/slide_25_img_462.png)

![Image 463](images/slide_25_img_463.png)

![Image 464](images/slide_25_img_464.png)

![Image 465](images/slide_25_img_465.png)

![Image 466](images/slide_25_img_466.png)

![Image 467](images/slide_25_img_467.png)

![Image 468](images/slide_25_img_468.png)

![Image 469](images/slide_25_img_469.png)

![Image 470](images/slide_25_img_470.png)

![Image 471](images/slide_25_img_471.png)

![Image 472](images/slide_25_img_472.png)

![Image 473](images/slide_25_img_473.png)

![Image 474](images/slide_25_img_474.png)

![Image 475](images/slide_25_img_475.png)

![Image 476](images/slide_25_img_476.png)

![Image 477](images/slide_25_img_477.png)

![Image 478](images/slide_25_img_478.png)

![Image 479](images/slide_25_img_479.png)

![Image 480](images/slide_25_img_480.png)

![Image 481](images/slide_25_img_481.png)

![Image 482](images/slide_25_img_482.png)

![Image 483](images/slide_25_img_483.png)

![Image 484](images/slide_25_img_484.png)

![Image 485](images/slide_25_img_485.png)

![Image 486](images/slide_25_img_486.png)

![Image 487](images/slide_25_img_487.png)

![Image 488](images/slide_25_img_488.png)

![Image 489](images/slide_25_img_489.png)

![Image 490](images/slide_25_img_490.png)

![Image 491](images/slide_25_img_491.png)

![Image 492](images/slide_25_img_492.png)

![Image 493](images/slide_25_img_493.png)

![Image 494](images/slide_25_img_494.png)

![Image 495](images/slide_25_img_495.png)

![Image 496](images/slide_25_img_496.png)

![Image 497](images/slide_25_img_497.png)

![Image 498](images/slide_25_img_498.png)

![Image 499](images/slide_25_img_499.png)

![Image 500](images/slide_25_img_500.png)

![Image 501](images/slide_25_img_501.png)

![Image 502](images/slide_25_img_502.png)

![Image 503](images/slide_25_img_503.png)

![Image 504](images/slide_25_img_504.png)

![Image 505](images/slide_25_img_505.png)

![Image 506](images/slide_25_img_506.png)

![Image 507](images/slide_25_img_507.png)

![Image 508](images/slide_25_img_508.png)

![Image 509](images/slide_25_img_509.png)

![Image 510](images/slide_25_img_510.png)

![Image 511](images/slide_25_img_511.png)

![Image 512](images/slide_25_img_512.png)

![Image 513](images/slide_25_img_513.png)

![Image 514](images/slide_25_img_514.png)

![Image 515](images/slide_25_img_515.png)

![Image 516](images/slide_25_img_516.png)

![Image 517](images/slide_25_img_517.png)

![Image 518](images/slide_25_img_518.png)

![Image 519](images/slide_25_img_519.png)

![Image 520](images/slide_25_img_520.png)

![Image 521](images/slide_25_img_521.png)

![Image 522](images/slide_25_img_522.png)

![Image 523](images/slide_25_img_523.png)

![Image 524](images/slide_25_img_524.png)

![Image 525](images/slide_25_img_525.png)

![Image 526](images/slide_25_img_526.png)

![Image 527](images/slide_25_img_527.png)

![Image 528](images/slide_25_img_528.png)

![Image 529](images/slide_25_img_529.png)

![Image 530](images/slide_25_img_530.png)

![Image 531](images/slide_25_img_531.png)

![Image 532](images/slide_25_img_532.png)

![Image 533](images/slide_25_img_533.png)

![Image 534](images/slide_25_img_534.png)

![Image 535](images/slide_25_img_535.png)

![Image 536](images/slide_25_img_536.png)

![Image 537](images/slide_25_img_537.png)

![Image 538](images/slide_25_img_538.png)

![Image 539](images/slide_25_img_539.png)

![Image 540](images/slide_25_img_540.png)

![Image 541](images/slide_25_img_541.png)

![Image 542](images/slide_25_img_542.png)

![Image 543](images/slide_25_img_543.png)

![Image 544](images/slide_25_img_544.png)

![Image 545](images/slide_25_img_545.png)

![Image 546](images/slide_25_img_546.png)

![Image 547](images/slide_25_img_547.png)

![Image 548](images/slide_25_img_548.png)

![Image 549](images/slide_25_img_549.png)

![Image 550](images/slide_25_img_550.png)

![Image 551](images/slide_25_img_551.png)

![Image 552](images/slide_25_img_552.png)

![Image 553](images/slide_25_img_553.png)

![Image 554](images/slide_25_img_554.png)

![Image 555](images/slide_25_img_555.png)

![Image 556](images/slide_25_img_556.png)

![Image 557](images/slide_25_img_557.png)

![Image 558](images/slide_25_img_558.png)

![Image 559](images/slide_25_img_559.png)

![Image 560](images/slide_25_img_560.png)

![Image 561](images/slide_25_img_561.png)

![Image 562](images/slide_25_img_562.png)

![Image 563](images/slide_25_img_563.png)

![Image 564](images/slide_25_img_564.png)

![Image 565](images/slide_25_img_565.png)

![Image 566](images/slide_25_img_566.png)

![Image 567](images/slide_25_img_567.png)

---

## Slide 26

Convolutional Filtering
Find
(kernel)
- Matched areas 
are amplified
- Objects are 
detected via 
convolution
- We have 
extracted the 
local features

![Image 1](images/slide_26_img_1.png)

![Image 2](images/slide_26_img_2.png)

![Image 3](images/slide_26_img_3.png)

![Image 4](images/slide_26_img_4.png)

![Image 5](images/slide_26_img_5.png)

![Image 6](images/slide_26_img_6.png)

![Image 7](images/slide_26_img_7.png)

![Image 8](images/slide_26_img_8.png)

![Image 9](images/slide_26_img_9.png)

![Image 10](images/slide_26_img_10.png)

![Image 11](images/slide_26_img_11.png)

![Image 12](images/slide_26_img_12.png)

![Image 13](images/slide_26_img_13.png)

![Image 14](images/slide_26_img_14.png)

![Image 15](images/slide_26_img_15.png)

![Image 16](images/slide_26_img_16.png)

![Image 17](images/slide_26_img_17.png)

![Image 18](images/slide_26_img_18.png)

![Image 19](images/slide_26_img_19.png)

![Image 20](images/slide_26_img_20.png)

![Image 21](images/slide_26_img_21.png)

![Image 22](images/slide_26_img_22.png)

![Image 23](images/slide_26_img_23.png)

![Image 24](images/slide_26_img_24.png)

![Image 25](images/slide_26_img_25.png)

![Image 26](images/slide_26_img_26.png)

![Image 27](images/slide_26_img_27.png)

![Image 28](images/slide_26_img_28.png)

![Image 29](images/slide_26_img_29.png)

![Image 30](images/slide_26_img_30.png)

![Image 31](images/slide_26_img_31.png)

![Image 32](images/slide_26_img_32.png)

![Image 33](images/slide_26_img_33.png)

![Image 34](images/slide_26_img_34.png)

![Image 35](images/slide_26_img_35.png)

![Image 36](images/slide_26_img_36.png)

![Image 37](images/slide_26_img_37.png)

![Image 38](images/slide_26_img_38.png)

![Image 39](images/slide_26_img_39.png)

![Image 40](images/slide_26_img_40.png)

![Image 41](images/slide_26_img_41.png)

![Image 42](images/slide_26_img_42.png)

![Image 43](images/slide_26_img_43.png)

![Image 44](images/slide_26_img_44.png)

![Image 45](images/slide_26_img_45.png)

![Image 46](images/slide_26_img_46.png)

![Image 47](images/slide_26_img_47.png)

![Image 48](images/slide_26_img_48.png)

![Image 49](images/slide_26_img_49.png)

![Image 50](images/slide_26_img_50.png)

![Image 51](images/slide_26_img_51.png)

![Image 52](images/slide_26_img_52.png)

![Image 53](images/slide_26_img_53.png)

![Image 54](images/slide_26_img_54.png)

![Image 55](images/slide_26_img_55.png)

![Image 56](images/slide_26_img_56.png)

![Image 57](images/slide_26_img_57.png)

![Image 58](images/slide_26_img_58.png)

![Image 59](images/slide_26_img_59.png)

![Image 60](images/slide_26_img_60.png)

![Image 61](images/slide_26_img_61.png)

![Image 62](images/slide_26_img_62.png)

![Image 63](images/slide_26_img_63.png)

![Image 64](images/slide_26_img_64.png)

![Image 65](images/slide_26_img_65.png)

![Image 66](images/slide_26_img_66.png)

![Image 67](images/slide_26_img_67.png)

![Image 68](images/slide_26_img_68.png)

![Image 69](images/slide_26_img_69.png)

![Image 70](images/slide_26_img_70.png)

![Image 71](images/slide_26_img_71.png)

![Image 72](images/slide_26_img_72.png)

![Image 73](images/slide_26_img_73.png)

![Image 74](images/slide_26_img_74.png)

![Image 75](images/slide_26_img_75.png)

![Image 76](images/slide_26_img_76.png)

![Image 77](images/slide_26_img_77.png)

![Image 78](images/slide_26_img_78.png)

![Image 79](images/slide_26_img_79.png)

![Image 80](images/slide_26_img_80.png)

![Image 81](images/slide_26_img_81.png)

![Image 82](images/slide_26_img_82.png)

![Image 83](images/slide_26_img_83.png)

![Image 84](images/slide_26_img_84.png)

![Image 85](images/slide_26_img_85.png)

![Image 86](images/slide_26_img_86.png)

![Image 87](images/slide_26_img_87.png)

![Image 88](images/slide_26_img_88.png)

![Image 89](images/slide_26_img_89.png)

![Image 90](images/slide_26_img_90.png)

![Image 91](images/slide_26_img_91.png)

![Image 92](images/slide_26_img_92.png)

![Image 93](images/slide_26_img_93.png)

![Image 94](images/slide_26_img_94.png)

![Image 95](images/slide_26_img_95.png)

![Image 96](images/slide_26_img_96.png)

![Image 97](images/slide_26_img_97.png)

![Image 98](images/slide_26_img_98.png)

![Image 99](images/slide_26_img_99.png)

![Image 100](images/slide_26_img_100.png)

![Image 101](images/slide_26_img_101.png)

![Image 102](images/slide_26_img_102.png)

![Image 103](images/slide_26_img_103.png)

![Image 104](images/slide_26_img_104.png)

![Image 105](images/slide_26_img_105.png)

![Image 106](images/slide_26_img_106.png)

![Image 107](images/slide_26_img_107.png)

![Image 108](images/slide_26_img_108.png)

![Image 109](images/slide_26_img_109.png)

![Image 110](images/slide_26_img_110.png)

![Image 111](images/slide_26_img_111.png)

![Image 112](images/slide_26_img_112.png)

![Image 113](images/slide_26_img_113.png)

![Image 114](images/slide_26_img_114.png)

![Image 115](images/slide_26_img_115.png)

![Image 116](images/slide_26_img_116.png)

![Image 117](images/slide_26_img_117.png)

![Image 118](images/slide_26_img_118.png)

![Image 119](images/slide_26_img_119.png)

![Image 120](images/slide_26_img_120.png)

![Image 121](images/slide_26_img_121.png)

![Image 122](images/slide_26_img_122.png)

![Image 123](images/slide_26_img_123.png)

![Image 124](images/slide_26_img_124.png)

![Image 125](images/slide_26_img_125.png)

![Image 126](images/slide_26_img_126.png)

![Image 127](images/slide_26_img_127.png)

![Image 128](images/slide_26_img_128.png)

![Image 129](images/slide_26_img_129.png)

![Image 130](images/slide_26_img_130.png)

![Image 131](images/slide_26_img_131.png)

![Image 132](images/slide_26_img_132.png)

![Image 133](images/slide_26_img_133.png)

![Image 134](images/slide_26_img_134.png)

![Image 135](images/slide_26_img_135.png)

![Image 136](images/slide_26_img_136.png)

![Image 137](images/slide_26_img_137.png)

![Image 138](images/slide_26_img_138.png)

![Image 139](images/slide_26_img_139.png)

![Image 140](images/slide_26_img_140.png)

![Image 141](images/slide_26_img_141.png)

![Image 142](images/slide_26_img_142.png)

![Image 143](images/slide_26_img_143.png)

![Image 144](images/slide_26_img_144.png)

![Image 145](images/slide_26_img_145.png)

![Image 146](images/slide_26_img_146.png)

![Image 147](images/slide_26_img_147.png)

![Image 148](images/slide_26_img_148.png)

![Image 149](images/slide_26_img_149.png)

![Image 150](images/slide_26_img_150.png)

![Image 151](images/slide_26_img_151.png)

![Image 152](images/slide_26_img_152.png)

![Image 153](images/slide_26_img_153.png)

![Image 154](images/slide_26_img_154.png)

![Image 155](images/slide_26_img_155.png)

![Image 156](images/slide_26_img_156.png)

![Image 157](images/slide_26_img_157.png)

![Image 158](images/slide_26_img_158.png)

![Image 159](images/slide_26_img_159.png)

![Image 160](images/slide_26_img_160.png)

![Image 161](images/slide_26_img_161.png)

![Image 162](images/slide_26_img_162.png)

![Image 163](images/slide_26_img_163.png)

![Image 164](images/slide_26_img_164.png)

![Image 165](images/slide_26_img_165.png)

![Image 166](images/slide_26_img_166.png)

![Image 167](images/slide_26_img_167.png)

![Image 168](images/slide_26_img_168.png)

![Image 169](images/slide_26_img_169.png)

![Image 170](images/slide_26_img_170.png)

![Image 171](images/slide_26_img_171.png)

![Image 172](images/slide_26_img_172.png)

![Image 173](images/slide_26_img_173.png)

![Image 174](images/slide_26_img_174.png)

![Image 175](images/slide_26_img_175.png)

![Image 176](images/slide_26_img_176.png)

![Image 177](images/slide_26_img_177.png)

![Image 178](images/slide_26_img_178.png)

![Image 179](images/slide_26_img_179.png)

![Image 180](images/slide_26_img_180.png)

![Image 181](images/slide_26_img_181.png)

![Image 182](images/slide_26_img_182.png)

![Image 183](images/slide_26_img_183.png)

![Image 184](images/slide_26_img_184.png)

![Image 185](images/slide_26_img_185.png)

![Image 186](images/slide_26_img_186.png)

![Image 187](images/slide_26_img_187.png)

![Image 188](images/slide_26_img_188.png)

![Image 189](images/slide_26_img_189.png)

![Image 190](images/slide_26_img_190.png)

![Image 191](images/slide_26_img_191.png)

![Image 192](images/slide_26_img_192.png)

![Image 193](images/slide_26_img_193.png)

![Image 194](images/slide_26_img_194.png)

![Image 195](images/slide_26_img_195.png)

![Image 196](images/slide_26_img_196.png)

![Image 197](images/slide_26_img_197.png)

![Image 198](images/slide_26_img_198.png)

![Image 199](images/slide_26_img_199.png)

![Image 200](images/slide_26_img_200.png)

![Image 201](images/slide_26_img_201.png)

![Image 202](images/slide_26_img_202.png)

![Image 203](images/slide_26_img_203.png)

![Image 204](images/slide_26_img_204.png)

![Image 205](images/slide_26_img_205.png)

![Image 206](images/slide_26_img_206.png)

![Image 207](images/slide_26_img_207.png)

![Image 208](images/slide_26_img_208.png)

![Image 209](images/slide_26_img_209.png)

![Image 210](images/slide_26_img_210.png)

![Image 211](images/slide_26_img_211.png)

![Image 212](images/slide_26_img_212.png)

![Image 213](images/slide_26_img_213.png)

![Image 214](images/slide_26_img_214.png)

![Image 215](images/slide_26_img_215.png)

![Image 216](images/slide_26_img_216.png)

![Image 217](images/slide_26_img_217.png)

![Image 218](images/slide_26_img_218.png)

![Image 219](images/slide_26_img_219.png)

![Image 220](images/slide_26_img_220.png)

![Image 221](images/slide_26_img_221.png)

![Image 222](images/slide_26_img_222.png)

![Image 223](images/slide_26_img_223.png)

![Image 224](images/slide_26_img_224.png)

![Image 225](images/slide_26_img_225.png)

![Image 226](images/slide_26_img_226.png)

![Image 227](images/slide_26_img_227.png)

![Image 228](images/slide_26_img_228.png)

![Image 229](images/slide_26_img_229.png)

![Image 230](images/slide_26_img_230.png)

![Image 231](images/slide_26_img_231.png)

![Image 232](images/slide_26_img_232.png)

![Image 233](images/slide_26_img_233.png)

![Image 234](images/slide_26_img_234.png)

![Image 235](images/slide_26_img_235.png)

![Image 236](images/slide_26_img_236.png)

![Image 237](images/slide_26_img_237.png)

![Image 238](images/slide_26_img_238.png)

![Image 239](images/slide_26_img_239.png)

![Image 240](images/slide_26_img_240.png)

![Image 241](images/slide_26_img_241.png)

![Image 242](images/slide_26_img_242.png)

![Image 243](images/slide_26_img_243.png)

![Image 244](images/slide_26_img_244.png)

![Image 245](images/slide_26_img_245.png)

![Image 246](images/slide_26_img_246.png)

![Image 247](images/slide_26_img_247.png)

![Image 248](images/slide_26_img_248.png)

![Image 249](images/slide_26_img_249.png)

![Image 250](images/slide_26_img_250.png)

![Image 251](images/slide_26_img_251.png)

![Image 252](images/slide_26_img_252.png)

![Image 253](images/slide_26_img_253.png)

![Image 254](images/slide_26_img_254.png)

![Image 255](images/slide_26_img_255.png)

![Image 256](images/slide_26_img_256.png)

![Image 257](images/slide_26_img_257.png)

![Image 258](images/slide_26_img_258.png)

![Image 259](images/slide_26_img_259.png)

![Image 260](images/slide_26_img_260.png)

![Image 261](images/slide_26_img_261.png)

![Image 262](images/slide_26_img_262.png)

![Image 263](images/slide_26_img_263.png)

![Image 264](images/slide_26_img_264.png)

![Image 265](images/slide_26_img_265.png)

![Image 266](images/slide_26_img_266.png)

![Image 267](images/slide_26_img_267.png)

![Image 268](images/slide_26_img_268.png)

![Image 269](images/slide_26_img_269.png)

![Image 270](images/slide_26_img_270.png)

![Image 271](images/slide_26_img_271.png)

![Image 272](images/slide_26_img_272.png)

![Image 273](images/slide_26_img_273.png)

![Image 274](images/slide_26_img_274.png)

![Image 275](images/slide_26_img_275.png)

![Image 276](images/slide_26_img_276.png)

![Image 277](images/slide_26_img_277.png)

![Image 278](images/slide_26_img_278.png)

![Image 279](images/slide_26_img_279.png)

![Image 280](images/slide_26_img_280.png)

![Image 281](images/slide_26_img_281.png)

![Image 282](images/slide_26_img_282.png)

![Image 283](images/slide_26_img_283.png)

![Image 284](images/slide_26_img_284.png)

![Image 285](images/slide_26_img_285.png)

![Image 286](images/slide_26_img_286.png)

![Image 287](images/slide_26_img_287.png)

![Image 288](images/slide_26_img_288.png)

![Image 289](images/slide_26_img_289.png)

![Image 290](images/slide_26_img_290.png)

![Image 291](images/slide_26_img_291.png)

![Image 292](images/slide_26_img_292.png)

![Image 293](images/slide_26_img_293.png)

![Image 294](images/slide_26_img_294.png)

![Image 295](images/slide_26_img_295.png)

![Image 296](images/slide_26_img_296.png)

![Image 297](images/slide_26_img_297.png)

![Image 298](images/slide_26_img_298.png)

![Image 299](images/slide_26_img_299.png)

![Image 300](images/slide_26_img_300.png)

![Image 301](images/slide_26_img_301.png)

![Image 302](images/slide_26_img_302.png)

![Image 303](images/slide_26_img_303.png)

![Image 304](images/slide_26_img_304.png)

![Image 305](images/slide_26_img_305.png)

![Image 306](images/slide_26_img_306.png)

![Image 307](images/slide_26_img_307.png)

![Image 308](images/slide_26_img_308.png)

![Image 309](images/slide_26_img_309.png)

![Image 310](images/slide_26_img_310.png)

![Image 311](images/slide_26_img_311.png)

![Image 312](images/slide_26_img_312.png)

![Image 313](images/slide_26_img_313.png)

![Image 314](images/slide_26_img_314.png)

![Image 315](images/slide_26_img_315.png)

![Image 316](images/slide_26_img_316.png)

![Image 317](images/slide_26_img_317.png)

![Image 318](images/slide_26_img_318.png)

![Image 319](images/slide_26_img_319.png)

![Image 320](images/slide_26_img_320.png)

![Image 321](images/slide_26_img_321.png)

![Image 322](images/slide_26_img_322.png)

![Image 323](images/slide_26_img_323.png)

![Image 324](images/slide_26_img_324.png)

![Image 325](images/slide_26_img_325.png)

![Image 326](images/slide_26_img_326.png)

![Image 327](images/slide_26_img_327.png)

![Image 328](images/slide_26_img_328.png)

![Image 329](images/slide_26_img_329.png)

![Image 330](images/slide_26_img_330.png)

![Image 331](images/slide_26_img_331.png)

![Image 332](images/slide_26_img_332.png)

![Image 333](images/slide_26_img_333.png)

![Image 334](images/slide_26_img_334.png)

![Image 335](images/slide_26_img_335.png)

![Image 336](images/slide_26_img_336.png)

![Image 337](images/slide_26_img_337.png)

![Image 338](images/slide_26_img_338.png)

![Image 339](images/slide_26_img_339.png)

![Image 340](images/slide_26_img_340.png)

![Image 341](images/slide_26_img_341.png)

![Image 342](images/slide_26_img_342.png)

![Image 343](images/slide_26_img_343.png)

![Image 344](images/slide_26_img_344.png)

![Image 345](images/slide_26_img_345.png)

![Image 346](images/slide_26_img_346.png)

![Image 347](images/slide_26_img_347.png)

![Image 348](images/slide_26_img_348.png)

![Image 349](images/slide_26_img_349.png)

![Image 350](images/slide_26_img_350.png)

![Image 351](images/slide_26_img_351.png)

![Image 352](images/slide_26_img_352.png)

![Image 353](images/slide_26_img_353.png)

![Image 354](images/slide_26_img_354.png)

![Image 355](images/slide_26_img_355.png)

![Image 356](images/slide_26_img_356.png)

![Image 357](images/slide_26_img_357.png)

![Image 358](images/slide_26_img_358.png)

![Image 359](images/slide_26_img_359.png)

![Image 360](images/slide_26_img_360.png)

![Image 361](images/slide_26_img_361.png)

![Image 362](images/slide_26_img_362.png)

![Image 363](images/slide_26_img_363.png)

![Image 364](images/slide_26_img_364.png)

![Image 365](images/slide_26_img_365.png)

![Image 366](images/slide_26_img_366.png)

![Image 367](images/slide_26_img_367.png)

![Image 368](images/slide_26_img_368.png)

![Image 369](images/slide_26_img_369.png)

![Image 370](images/slide_26_img_370.png)

![Image 371](images/slide_26_img_371.png)

![Image 372](images/slide_26_img_372.png)

![Image 373](images/slide_26_img_373.png)

![Image 374](images/slide_26_img_374.png)

![Image 375](images/slide_26_img_375.png)

![Image 376](images/slide_26_img_376.png)

![Image 377](images/slide_26_img_377.png)

![Image 378](images/slide_26_img_378.png)

![Image 379](images/slide_26_img_379.png)

![Image 380](images/slide_26_img_380.png)

![Image 381](images/slide_26_img_381.png)

![Image 382](images/slide_26_img_382.png)

![Image 383](images/slide_26_img_383.png)

![Image 384](images/slide_26_img_384.png)

![Image 385](images/slide_26_img_385.png)

![Image 386](images/slide_26_img_386.png)

![Image 387](images/slide_26_img_387.png)

![Image 388](images/slide_26_img_388.png)

![Image 389](images/slide_26_img_389.png)

![Image 390](images/slide_26_img_390.png)

![Image 391](images/slide_26_img_391.png)

![Image 392](images/slide_26_img_392.png)

![Image 393](images/slide_26_img_393.png)

![Image 394](images/slide_26_img_394.png)

![Image 395](images/slide_26_img_395.png)

![Image 396](images/slide_26_img_396.png)

![Image 397](images/slide_26_img_397.png)

![Image 398](images/slide_26_img_398.png)

![Image 399](images/slide_26_img_399.png)

![Image 400](images/slide_26_img_400.png)

![Image 401](images/slide_26_img_401.png)

![Image 402](images/slide_26_img_402.png)

![Image 403](images/slide_26_img_403.png)

![Image 404](images/slide_26_img_404.png)

![Image 405](images/slide_26_img_405.png)

![Image 406](images/slide_26_img_406.png)

![Image 407](images/slide_26_img_407.png)

![Image 408](images/slide_26_img_408.png)

![Image 409](images/slide_26_img_409.png)

![Image 410](images/slide_26_img_410.png)

![Image 411](images/slide_26_img_411.png)

![Image 412](images/slide_26_img_412.png)

![Image 413](images/slide_26_img_413.png)

![Image 414](images/slide_26_img_414.png)

![Image 415](images/slide_26_img_415.png)

![Image 416](images/slide_26_img_416.png)

![Image 417](images/slide_26_img_417.png)

![Image 418](images/slide_26_img_418.png)

![Image 419](images/slide_26_img_419.png)

![Image 420](images/slide_26_img_420.png)

![Image 421](images/slide_26_img_421.png)

![Image 422](images/slide_26_img_422.png)

![Image 423](images/slide_26_img_423.png)

![Image 424](images/slide_26_img_424.png)

![Image 425](images/slide_26_img_425.png)

![Image 426](images/slide_26_img_426.png)

![Image 427](images/slide_26_img_427.png)

![Image 428](images/slide_26_img_428.png)

![Image 429](images/slide_26_img_429.png)

![Image 430](images/slide_26_img_430.png)

![Image 431](images/slide_26_img_431.png)

![Image 432](images/slide_26_img_432.png)

![Image 433](images/slide_26_img_433.png)

![Image 434](images/slide_26_img_434.png)

![Image 435](images/slide_26_img_435.png)

![Image 436](images/slide_26_img_436.png)

![Image 437](images/slide_26_img_437.png)

![Image 438](images/slide_26_img_438.png)

![Image 439](images/slide_26_img_439.png)

![Image 440](images/slide_26_img_440.png)

![Image 441](images/slide_26_img_441.png)

![Image 442](images/slide_26_img_442.png)

![Image 443](images/slide_26_img_443.png)

![Image 444](images/slide_26_img_444.png)

![Image 445](images/slide_26_img_445.png)

![Image 446](images/slide_26_img_446.png)

![Image 447](images/slide_26_img_447.png)

![Image 448](images/slide_26_img_448.png)

![Image 449](images/slide_26_img_449.png)

![Image 450](images/slide_26_img_450.png)

![Image 451](images/slide_26_img_451.png)

![Image 452](images/slide_26_img_452.png)

![Image 453](images/slide_26_img_453.png)

![Image 454](images/slide_26_img_454.png)

![Image 455](images/slide_26_img_455.png)

![Image 456](images/slide_26_img_456.png)

![Image 457](images/slide_26_img_457.png)

![Image 458](images/slide_26_img_458.png)

![Image 459](images/slide_26_img_459.png)

![Image 460](images/slide_26_img_460.png)

![Image 461](images/slide_26_img_461.png)

![Image 462](images/slide_26_img_462.png)

![Image 463](images/slide_26_img_463.png)

![Image 464](images/slide_26_img_464.png)

![Image 465](images/slide_26_img_465.png)

![Image 466](images/slide_26_img_466.png)

![Image 467](images/slide_26_img_467.png)

![Image 468](images/slide_26_img_468.png)

![Image 469](images/slide_26_img_469.png)

![Image 470](images/slide_26_img_470.png)

![Image 471](images/slide_26_img_471.png)

![Image 472](images/slide_26_img_472.png)

![Image 473](images/slide_26_img_473.png)

![Image 474](images/slide_26_img_474.png)

![Image 475](images/slide_26_img_475.png)

![Image 476](images/slide_26_img_476.png)

![Image 477](images/slide_26_img_477.png)

![Image 478](images/slide_26_img_478.png)

![Image 479](images/slide_26_img_479.png)

![Image 480](images/slide_26_img_480.png)

![Image 481](images/slide_26_img_481.png)

![Image 482](images/slide_26_img_482.png)

![Image 483](images/slide_26_img_483.png)

![Image 484](images/slide_26_img_484.png)

![Image 485](images/slide_26_img_485.png)

![Image 486](images/slide_26_img_486.png)

![Image 487](images/slide_26_img_487.png)

![Image 488](images/slide_26_img_488.png)

![Image 489](images/slide_26_img_489.png)

![Image 490](images/slide_26_img_490.png)

![Image 491](images/slide_26_img_491.png)

![Image 492](images/slide_26_img_492.png)

![Image 493](images/slide_26_img_493.png)

![Image 494](images/slide_26_img_494.png)

![Image 495](images/slide_26_img_495.png)

![Image 496](images/slide_26_img_496.png)

![Image 497](images/slide_26_img_497.png)

![Image 498](images/slide_26_img_498.png)

![Image 499](images/slide_26_img_499.png)

![Image 500](images/slide_26_img_500.png)

![Image 501](images/slide_26_img_501.png)

![Image 502](images/slide_26_img_502.png)

![Image 503](images/slide_26_img_503.png)

![Image 504](images/slide_26_img_504.png)

![Image 505](images/slide_26_img_505.png)

![Image 506](images/slide_26_img_506.png)

![Image 507](images/slide_26_img_507.png)

![Image 508](images/slide_26_img_508.png)

![Image 509](images/slide_26_img_509.png)

![Image 510](images/slide_26_img_510.png)

![Image 511](images/slide_26_img_511.png)

![Image 512](images/slide_26_img_512.png)

![Image 513](images/slide_26_img_513.png)

![Image 514](images/slide_26_img_514.png)

![Image 515](images/slide_26_img_515.png)

![Image 516](images/slide_26_img_516.png)

![Image 517](images/slide_26_img_517.png)

![Image 518](images/slide_26_img_518.png)

![Image 519](images/slide_26_img_519.png)

![Image 520](images/slide_26_img_520.png)

![Image 521](images/slide_26_img_521.png)

![Image 522](images/slide_26_img_522.png)

![Image 523](images/slide_26_img_523.png)

![Image 524](images/slide_26_img_524.png)

![Image 525](images/slide_26_img_525.png)

![Image 526](images/slide_26_img_526.png)

![Image 527](images/slide_26_img_527.png)

![Image 528](images/slide_26_img_528.png)

![Image 529](images/slide_26_img_529.png)

![Image 530](images/slide_26_img_530.png)

![Image 531](images/slide_26_img_531.png)

![Image 532](images/slide_26_img_532.png)

![Image 533](images/slide_26_img_533.png)

![Image 534](images/slide_26_img_534.png)

![Image 535](images/slide_26_img_535.png)

![Image 536](images/slide_26_img_536.png)

![Image 537](images/slide_26_img_537.png)

![Image 538](images/slide_26_img_538.png)

![Image 539](images/slide_26_img_539.png)

![Image 540](images/slide_26_img_540.png)

![Image 541](images/slide_26_img_541.png)

![Image 542](images/slide_26_img_542.png)

![Image 543](images/slide_26_img_543.png)

![Image 544](images/slide_26_img_544.png)

![Image 545](images/slide_26_img_545.png)

![Image 546](images/slide_26_img_546.png)

![Image 547](images/slide_26_img_547.png)

![Image 548](images/slide_26_img_548.png)

![Image 549](images/slide_26_img_549.png)

![Image 550](images/slide_26_img_550.png)

![Image 551](images/slide_26_img_551.png)

![Image 552](images/slide_26_img_552.png)

![Image 553](images/slide_26_img_553.png)

![Image 554](images/slide_26_img_554.png)

![Image 555](images/slide_26_img_555.png)

![Image 556](images/slide_26_img_556.png)

![Image 557](images/slide_26_img_557.png)

![Image 558](images/slide_26_img_558.png)

![Image 559](images/slide_26_img_559.png)

![Image 560](images/slide_26_img_560.png)

![Image 561](images/slide_26_img_561.png)

![Image 562](images/slide_26_img_562.png)

![Image 563](images/slide_26_img_563.png)

![Image 564](images/slide_26_img_564.png)

![Image 565](images/slide_26_img_565.png)

![Image 566](images/slide_26_img_566.png)

![Image 567](images/slide_26_img_567.png)

---

## Slide 27

Convolution in Time Series
- Pattern matching with kernels
- Assumption: Local features are detected by a 
series of peaks
- Period of seasonality can be identified by the 
peaks of cross-correlations
- Holiday effects are also identified by the 
irregularity in cross-correlation peaks
- One time series may align (correlate) with a 
mixture of kernels
time (t)
Kernel w1
Feature
map c1
Kernel w2
Feature
map c2

---

## Slide 28

Convolutional Neural Networks
- Learn several kernels from the dataset and 
identify which kernels to be used at which time
- Convolution layer with 3 kernels (width=10)
where each wi is a vector of 10 parameters
- Max-pooling for most prominent local features
- Nonlinear function
- The matrix is then flattened to become a vector
time (t)
c1
c2
c3
Conv1D(kernel=3, width=10)
Max Pooling
c*
ReLU
Flatten
Output vector for prediction with MLP
ci = x →wi
c→= max [c1|c2|c3]
x'
x→= ReLU(c↑)
N.B. These kernels can 
learn any shape of trends

---

## Slide 29

CNNs for Computer Vision
Feature
maps
Pooled
feature map
Feature
maps
Pooled
feature map


x1
x2
x3
...
xN


Flattened
vector
MLP
Class y

![Image 1](images/slide_29_img_1.jpeg)

---

## Slide 30

3. Frequency-Domain Methods

---

## Slide 31

3.1 Spectral Density Analysis

---

## Slide 32

Spectral Density
- Describing a time series 
in terms of power 
distribution according to 
frequency components
- Assumption: Time 
series is a mixture of 
periodic sinusoid waves 
(seasonal patterns)
- Transforming a time 
series into spectrum 
(squared amplitude) in 
the frequency domain
Source: https://dibsmethodsmeetings.github.io/fourier-transforms/ 
Spectrum representation

![Image 1](images/slide_32_img_1.png)

---

## Slide 33

Wave as Complex Number
- A complex number a+îb consists of the real part 
a and the imaginary part b, where î = √-1
- Wave can be represented as a complex number
u(x, t) = A cos(kx −!t + ✓)
phase
θ
Y-Axis
X-Axis
This is at time t = 0
We are at the kth wave
The beginning of the kth wave
amplitude A
frequency
Euler’s formula
exp[ˆıω]
= cos ω +ˆı sin ω
θ
radius = 1
where
a+îb
A
θ
At t = 0 and k = 0
u(x, t) = u0e→ˆı(ωt→kx)
u0 = Aeˆıω

---

## Slide 34

Wave as 3D Spiral
- Wave can be seen as a 
counterclockwise spiral in 3D 
space, whose base plane are 
real and imaginary parts
- Conjugate of a wave is 
therefore a clockwise spiral
exp[ˆıωt] = cos ωt +ˆı sin ωt
real
part
imaginary
part
exp [ˆıωt]→= exp[→ˆıωt]
= cos ωt →ˆı sin ωt

![Image 1](images/slide_34_img_1.png)

---

## Slide 35

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
frequency (ω)
2π
4π
6π
8π
|F{x}(ω)|2

---

## Slide 36

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
frequency (ω)
2π
4π
6π
8π
Average
the squared
amplitudes
frequency
= 2π

---

## Slide 37

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
frequency (ω)
2π
4π
6π
8π
frequency
= 4π
Average
the squared
amplitudes

---

## Slide 38

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
frequency (ω)
2π
4π
6π
8π
frequency
= 6π
Average
the squared
amplitudes

---

## Slide 39

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
frequency (ω)
2π
4π
6π
8π
frequency
= 8π
Average
the squared
amplitudes

---

## Slide 40

Fourier Transform
- Fourier transform of a time series x(t) is a 
function of frequency ω that reflects how well 
x(t) aligns with the wave of frequency ω 
- The frequency is usually a multiple of 2π (i.e. 
one round of a circle in radian)
4="bCOvkAIDe+SKa+7XLx47J2/o7k=">A
CwXiclVFNb9QwEHXCV1m+tnDkMmIF2iK6ShAF
LqCKSohjkdi20jpdeR1n16odR/ak7MrKn+QE/
wYnzQG6XBjJ8tO8N2/smUWlpMk+RXFN27eu
n1n5+7g3v0HDx8Ndx+fOFNbLqbcKGPFswJU
sxRYlKnFVWML1Q4nRxcdTyp5fCOmnKb7ipRK
bZspSF5AxDaj78STXDFWfKf26AelgDbWBMjRZ
Ltgcv4AOsx7gH1OQGodNa7b+zS9H0olfQ0nT
QSqksce73w1Xgpjn3PegteGvhqVhXM7piGNjQ
sfMAzJrzl0Ahx/+w6pz2YdurM5oPR8k6QK2Q
dqDEenjeD78QXPDay1K5Io5N0uTCjPLEquR
DOgtRMV4xdsKWYBlkwLl/luAw08D5kcCmPDKR
G67J8VnmnNnoRlO0M3XWuTf6Lm9VYvM+8LK
saRcmvGhW1AjTQrhNyaQVHtQmAcSvDW4GvmGU
cw9IHYQjp9S9vg5PXk/Tt5ODrm9Hhp34cO+Qp
eUbGJCXvyCH5Qo7JlPDoY5RHOirjo1jGVWyv
pHU1zwhf0XsfwP5NcX</latexit>F{x}(ω) = x(t) →wave(ω, t)
=
! →
↑→
x(t) · exp[ˆıωt]↓dt
=
! →
↑→
x(t) · exp[↑ˆıωt] dt
time (t)
2π
4π
6π
8π
frequency
= 8π
Average
the squared
amplitudes
Spectrum

---

## Slide 41

Fast Fourier Transform (Cooley & Tukey, 1965)
def fft(x: array, N: length, s: stride):
    result := create_array(size=N)
    if N == 1:
        result[0] = x[0]
    else:
        result[0 : N/2] := fft(x, N/2, 2*s)       # x[0], x[2s], x[4s], ...
        result[N/2 : N] := fft(x[s:], N/2, 2*s)   # x[s], x[3s], x[5s], ...
        for k in range(N/2):
            p := result[k]
            q := result[k + N/2] * exp(-2 * π * î * k/N)
            result[k] := p + q
            result[k + N/2] := p - q
    return result
HOW TO RUN ==> fft(x, N, s=1) ==> Then compute the squared magnitude of each element
O(n log n) 
time 
complexity

---

## Slide 42

Combine
Fast Fourier Transform (Cooley & Tukey, 1965)
- Base case
- Recursive case
F(x[k]) = x[k]
6pr9R9ue2xl6GmpRBiJNMmvRC6GnkEKcBLTCrNYre/FK2uyOSozQpZceWkqu+Vu59Y/k3JWtQxL71IGFj5nvm+cmWkmLQfDX8+/cvXf/wcbDzqPHT54+6z5/cWSL0nAx5IUqzEnCr
FAyF0OUqMSJNoJliRLHyexLEz/+LoyVRX6Icy3ijE1ymUrO0LlG3SsNb+ET0IzhlDNV7dXRWTSLY6C0c7ouBJtAU8N4tV9Xg3pJXFU3Qu2op0BRZsICFWcaqBIpRltApwrKus204Bq
CbPaZQRq5GSK65Ju3qy6LD1vwVG3V7QDxYGqyBsQY+0djDqXtJxwctM5MgVszYKA41xQxKrkTdoaUVmvEZm4jIwZy5nuJqcZ8a3jPGNLCuJcjLzXFRXLrJ1niWM2Q9vbsca5Lha
VmH6MK5nrEkXOl4XSUgEW0BwbxtIjmruAONGul6BT5lbCbov0XFLCG+PvAqOBv1wu/h2/ve7ud2HRvkFXlN3pGQ7JBd8pUckCHhHvV+eL+83z7zf/p/PMl1fdazUtyw/yLfxwo4p
8=</latexit>
p = F[x[k]]
q = F[x[k + N
2 ]]
F[x[k]] = p + q →exp
!
↑ˆı2ωk
N
"
F[x[k + N
2 ]] = p ↑q →exp
!
↑ˆı2ωk
N
"
Time series x[1 … N]
FFT
FFT
p
q
N/2
texit 
sha1_
base6
4="jW1
YZNUt
zIzXsE
FxcZF
todf/
CRQ=">
ACJ
HicbVB
NSyNB
EO2J6
5qN7hr
16KXZ
IAiyYS
b4BV7
CevEk
CpsYSA
+hp1O
TNOmZ
abtrxD
Dkx3j
xr+zFg
+6yBy
/+Fjs
xh924D
woe71
VRVS/S
Slr0/
WevtP
Rh+eNK
+VNld
e3zl/X
qxmb
ZrkR0
BKZykw
n4haU
TKGFEh
V0tAG
eRAqu
otHp1L
+6AWN
lv7As
Yw4Y
NUxlJ
wdFKve
qLpHr
2mDGUC
ljK41
ZQpiL
H7jbIh
x4LJC
Wx4a
JoMC3p
aFKcO
8HIwRD
DXrXm
1/0Z6
HsSzEm
NzHR
q/5i/U
zkCaQ
oFLe2
G/gaw4
IblEL
BpMJyC
5qLER
9A19G
Uu5vCY
vbkhO
4pU/j
zLhKk
c7Uvy
cKnlg7
TiLXm
XAc2kV
vKv7P
6+YH
4eFTHW
OkIq3
RXGuKG
Z0mhj
tSwMC
1dgRLo
x0t1I
x5C4Sd
LlWXA
jB4sv
vSbtRD
w7rB5
f7teb
3eRxls
k2+kl
0SkCPS
JGfkg
rSIH
fkJ3k
T969
+D9v6
8tZa8
+cwW+
Qfeyt
dMqQa
</late
xit>
p + q →exp
!
↑ˆı2ωk
N
"
BxkvrnY05WKrQ+BeIpJCr/BKCcE=">ACJHicbVDJS
gNBEO1xN25Rj14ag+DFMCNu4EX04kUjArpIfR0apIm
PTNtd40YhnyMF3/FiwcXPHjxW+zEHNweFDzeq6KqXqSV
tOj797I6Nj4xOTUdGlmdm5+oby4dGz3AioiUxl5ir
iFpRMoYSFVxpAzyJFxGnaO+f3kDxsosPceuhjDhrV
TGUnB0UqO8r+kGvaYMZQKWMrjVlCmIsb5BWZtjwWSPs
thwUWwyLWmnV5w4wchWG8NGueJX/QHoXxIMSYUMcdo
v7BmJvIEUhSKW1sPfI1hwQ1KoaBXYrkFzUWHt6DuaMr
dTWExeLJH15zSpHFmXKVIB+r3iYIn1naTyHUmHNv2t9
cX/PqOcZ7YSFTnSOk4mtRnCuKGe0nRpvSgEDVdYQLI9
2tVLS5iwRdriUXQvD75b/kYrMa7FS3z7YqB4fDOKbIC
lkl6yQgu+SAHJNTUiOC3JEH8kSevXv0Xv13r5aR7zh
zDL5Ae/jE2CqpBw=</latexit>
p →q ↑exp
!
→ˆı2ωk
N
"

---

## Slide 43

Spectogram
- Heatmap is used 
for visual 
representation of 
the spectrum of 
frequency
- Time series is 
tokenized into 
equal chunks (w.r.t. 
window size) and 
analyzed with FFT
- Seasonality and 
holiday effects are 
present
- Good for stationary 
signals e.g. long-
term climate data
Source: https://en.wikipedia.org/wiki/Spectrogram 

![Image 1](images/slide_43_img_1.jpeg)

---

## Slide 44

Asynchronous Speech Recognition
- Spectrogram is used 
as an input picture for 
CNN-based models
- Local features are 
changes in specific 
frequency ranges
- Conv2D is usually 
employed
Source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf 

![Image 1](images/slide_44_img_1.jpeg)

---

## Slide 45

3.2 Wavelet Analysis

---

## Slide 46

Wavelet Transform
- Wavelet transform of x(t) is a function of frequency 
a and position b that reflects how well x(t) aligns 
with the wavelet Ψ(a,b,t) 
where Morlet wavelet (1984) is used here:
- In practice, we need a low-pass filter Φ(t) to 
eliminate the inherent noise
time (t)
Morlet wavelet Ψ(a,b,t)
!(a, b, t) = cos a(t →b) exp
!
(t →b)2"
W{x}(a, b) = x(t) →!(a, b, t)
=
! →
↑→
x(t) · cos a(t ↑b) exp
"
(t ↑b)2#
dt

![Image 1](images/slide_46_img_1.png)

---

## Slide 47

Wavelet Transform in Time Series
- Pattern matching with wavelet
- Assumption: Local features are detected by a 
series of peaks
- Period of seasonality can be identified by the 
peaks of cross-correlations
- Holiday effects are also identified by the 
irregularity in cross-correlation peaks
- One time series may align (correlate) with a 
mixture of wavelets
time (t)
Morlet
wavelet
Feature
map

![Image 1](images/slide_47_img_1.jpeg)

---

## Slide 48

Fast Wavelet Transform (Mallat, 1989)
def fwt(x: array, N: length, h: low-pass filter, g: wavelet, L: resolution level)
    result := []
    arr := x
    for i = 1 to L:    # Resolution i
        row = []
        for k = 0 to N/2 – 1:
            coeff := sum(g[j] * arr[2*k - j] for j = 0 to len(g))
            row.append(coeff)
        result.insert(0, row)
        arr_next := []
        for k = 0 to N/2 – 1:
            coeff := sum(h[j] * arr[2*k - j] for j = 0 to len(h))
            arr_next.append(coeff)
        arr := arr_next
     result.insert(0, arr)
     return result
O(n log n) 
time 
complexity

---

## Slide 49

Fast Wavelet Transform (Mallat, 1989)
- Iterative procedure
Time series x[1 … N] of resolution i
g
Convolve with
wavelet g
x[…]
Resolution i+1
x[…]
h
Convolve with
low-pass filter h
Noise-reduced data
SCALOGRAM
R5
R4
R3
R2
R1

---

## Slide 50

Scalogram
- Heatmap is used for 
visual representation 
of the spectrum of 
frequency
- The entire time series 
is analyzed with FWT 
to extract peaks
- Both seasonality 
(frequency) and 
holiday effects 
(positions) are present
- Good for non-
stationary and noisy 
signals e.g. speech, 
EEG, and brain waves
Source: https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html 
- Peak reflects existence of 
a wavelet
- Horizontal ridge means 
time-consistent frequency
- Vertical ridge means 
sequence of constant 
frequency
- Separate ridges show a 
mixture of several tunes

![Image 1](images/slide_50_img_1.png)

---

## Slide 51

Heart Sound Classification via Wavelets
Lee, J.-A., & Kwak, K.-C. (2023). Heart Sound Classification Using Wavelet Analysis Approaches and Ensemble of Deep Learning 
Models. Applied Sciences, 13(21), 11942.

![Image 1](images/slide_51_img_1.jpeg)

---

## Slide 52

Brain Wave-to-Word Classification
- Scalogram is used 
as an input picture 
for CNN-based 
models
- Stimulations and 
background noises 
are preserved
- Conv2D is usually 
employed
Source: https://www.mathworks.com/company/user_stories/
ut-austin-researchers-convert-brain-signals-to-words-and-phrases-using-wavelets-and-deep-learning.html 

![Image 1](images/slide_52_img_1.jpeg)

---

## Slide 53

From Wavelet to JPEG
- Each wavelet of different resolutions 
extracts pixel changes in the image 
via Haar wavelet
- Compression becomes 
multiresolution extraction via fast 
wavelet transform
- Image can be reconstructed by 
combining these pixel changes

![Image 1](images/slide_53_img_1.jpeg)

![Image 2](images/slide_53_img_2.png)

---

## Slide 54

4. Transformer for Time Series

---

## Slide 55

Transformer Model (Vaswani et al., 2016)
- Sequence-to-sequence generation
- Translation: It learns how to produce a target 
sequence from a source sequence, given a very 
large dataset of sequence pairs
- Pros: It learns word collocations and phrase 
structures on the input and output sequences, 
and associates them cross-lingually in the table 
of translation alignments
- Cons: It consists of an expansive amount of 
neuron cells, and the training process can be 
quite time-consuming
TRANSFORMER
Source: sequence of words (prompt)
Target: sequence of words (response)
Who
is
the
current
president
of
the
US
The
president
of
the
US
is
Joe
Biden

---

## Slide 56

Scaled Dot-Product Attention
- Semantic similarity ⇒ search engine
- Query is compared against each key with dot product
- The more similar the key is to the query, the more 
weight its value will get
Query
Keys
Values
Weights
Scaled
Values
Combined
Result
RSKblxJBfuATh0yaYNzUzGJFMpw3ycGz/AnV/gxoUibsVMOxVtvRA4Oem5vjhoxKZrPxtz8wuLScm4lv7q2vrFZ2NquSx4JTGqYMy6aLpKE0YDUFWMNENBkO8y0nD7F6neGBAhKQ9u1DAkbR91A+pRjJSmnELr3qHwANqh4KHi0P
aR6rle3E80beMOVz/UXQJtOz+5iUS7ytCWke/EtGwlt1cwHTXRB3qAUyiaJXNUcBZYGSiCrKpO4cnucBz5JFCYISlblhmqdoyEopiRJG9HkoQI91GXtDQMkE9kOx6FkMB9zXSgx4U+gYIj9rcjRr6UQ9/VnemOclpLyf+0VqS8s3ZMgzB
SJMDjh7yIQR1XmijsUEGwYkMNEBZU7wpxDwmElc49r0Owpr8C+pHJeukZF0fFyvnWRw5sAv2wCGwCmogEtQBTWAwQN4AW/g3Xg0Xo0P43PcOmdknh3wp4yvb9gHsWU=</latexit>wi / ki · q
r =
N
X
i=1
wivi
w = Softmax(K ⇥q)
r = V> ⇥w
Simple
Form
Matrix
Form

---

## Slide 57

Mary
looks
this
word
up
Scaled Dot-Product Attention
- Semantic similarity ⇒ search engine
- Query is compared against each key with dot product
- The more similar the key is to the query, the more 
weight its value will get
Query
Keys
Values
Weights
Scaled
Values
Combined
Result
RSKblxJBfuATh0yaYNzUzGJFMpw3ycGz/AnV/gxoUibsVMOxVtvRA4Oem5vjhoxKZrPxtz8wuLScm4lv7q2vrFZ2NquSx4JTGqYMy6aLpKE0YDUFWMNENBkO8y0nD7F6neGBAhKQ9u1DAkbR91A+pRjJSmnELr3qHwANqh4KHi0P
aR6rle3E80beMOVz/UXQJtOz+5iUS7ytCWke/EtGwlt1cwHTXRB3qAUyiaJXNUcBZYGSiCrKpO4cnucBz5JFCYISlblhmqdoyEopiRJG9HkoQI91GXtDQMkE9kOx6FkMB9zXSgx4U+gYIj9rcjRr6UQ9/VnemOclpLyf+0VqS8s3ZMgzB
SJMDjh7yIQR1XmijsUEGwYkMNEBZU7wpxDwmElc49r0Owpr8C+pHJeukZF0fFyvnWRw5sAv2wCGwCmogEtQBTWAwQN4AW/g3Xg0Xo0P43PcOmdknh3wp4yvb9gHsWU=</latexit>wi / ki · q
r =
N
X
i=1
wivi
Simple
Form
w = Softmax(K ⇥q)
r = V> ⇥w
Matrix
Form
‘looks’
For word sequence, 
collocating words 
are semantically 
similar to each other
e.g. ‘looks ___ up’

---

## Slide 58

Self-Attention
- Scaled dot-product attention whose queries 
and keys are the same
- Collocations will have almost similar results
Mary
looks
this
word
up
Keys
Values
Mary
looks
this
word
up
Queries
Mary
looks
this
word
up
Mary
looks
this
word
up
Combined
Results
Matrix
Form
W = Softmax(K ⇥K>)
R = W ⇥V

---

## Slide 59

Cross-Attention
- Scaled dot-product attention whose queries 
are the target and whose keys are the source
- Collocation alignment via semantic similarity
Mary
looks
this
word
up
Keys
(source)
Values
แมรี
ค้นหา
คำ
นี้
Queries
(target)
Mary
looks
this
word
up
แมรี
ค้นหา
คำ
นี้
Combined
Results
Matrix
Form
W = Softmax(Q ⇥K>)
R = W ⇥V

---

## Slide 60

Multihead Attention
- Scaled dot-product attention has a drawback
- It recognizes only one type of word collocation
- If we assume more than one type of word 
collocation per sequence, then we have to combine 
multiple attention heads [default = 8 heads]
Scaled
dot-product
attention
Scaled
dot-product
attention
Queries
Keys
Values
LINEAR
LINEAR
CONCATENATION
LINEAR
Result
Mary
Poppins
looks
this
word
up
Mary
Poppins
looks
this
word
up
Mary
Poppins
looks
this
word
up
Mary
Poppins
looks
this
word
up
HEAD 1 (looks ___ up)
HEAD 2 (Mary Poppins)
Notation
Multihead
attention (n)
Q
K
V

---

## Slide 61

Informer
(Zhou et al., AAAI-2021)
Decoder
Outputs
Masked Multi-head
ProbSparse
Self-attention
Multi-head
Attention
Encoder
Inputs:    Xen
Concatenated Feature Map
Inputs:    Xde={Xtoken, X0}
0 0 0 0 0 0 0
Fully Connected Layer
Multi-head
ProbSparse
Self-attention
Multi-head
ProbSparse
Self-attention
Scalar
Stamp
T = t
T = t + Dx
L
d
Conv1d
L
d
Embedding
+
L
k
L
n-heads
Attention Block 1
Conv1d
MaxPool1d,
padding=2
L/2
k
L/2
n-heads
Attention Block 2
Conv1d
MaxPool1d,
padding=2
L/4
k
L/4
n-heads
Attention Block 3
L/4
d
Feature
Map
Overview
Encoder
Block
ProbSparse Self-attention Based on the proposed mea-
surement, we have the ProbSparse self-attention by allowing
each key to only attend to the u dominant queries:
A(Q, K, V) = Softmax(QK>
p
d
)V
,
(3)
where Q is a sparse matrix of the same size of q and it
only contains the Top-u queries under the sparsity measure-
ment M(q, K). Controlled by a constant sampling factor c,
we set u = c · ln LQ, which makes the ProbSparse self-
attention only need to calculate O(ln LQ) dot-product for
each query-key lookup and the layer memory usage main-
tains O(LK ln LQ). Under the multi-head perspective, this
i
diff
k
i
f
h

---

## Slide 62

Autoformer (Wu et al., NIPS-2021)
Input Data Mean
Auto-
Correlation
Series
Decomp
Trend-cyclical Init
Seasonal Init
Feed 
Forward
Series
Decomp
Auto-
Correlation
Series
Decomp
Auto-
Correlation
N x
M x
+
Encoder Input
Autoformer Decoder
Autoformer Encoder
+
Prediction
+
+
+
Feed 
Forward
Series
Decomp
+
Series
Decomp
+
+
+
Zero
To Predict
Zero
Data
Mean
Time
Series
Seasonal
Part
Trend
-cyclical
Part
K
V
Q
K
V
Q
K
V
Q
SeriesDecomp
Output = (Xs, Xt)
Xt = AvgPool(Padding(X))
Xs = X −Xt,
Autocorrelation
Block
…
L
Q
K
V
FFT
x
FFT
Inverse FFT
Time Delay Agg
Linear
Linear
Linear
Linear
Concat
LxC
SxC
SxC
LxCx2
LxC
LxC
LxCx2
Conjugate
LxCx2
LxC
Resize
Resize
LxC
Roll(     )
Roll(     )
Roll(     )
x
x
x
…
Fusion
Time
Delay
Top k
kxC
SoftMax
⌧1
⌧2
⌧k
R(⌧1)
R(⌧2)
R(⌧k)

![Image 1](images/slide_62_img_1.png)

![Image 2](images/slide_62_img_2.png)

---

## Slide 63

5. Conclusion

---

## Slide 64

Patterns in Time Series
- Trend: general direction over time
- Seasonality: repetitive patterns that occur at 
regular predictable intervals
- Holiday effects: irregular patterns caused by 
special calendar events
- Cycle: long-term repetitive patterns that occur 
at irregular intervals
time (t)
holiday effect
time (t)
trend
Cycle
1. Slow increase
2. Catastrophe
3. Rapid decline
season

---

## Slide 65

Time-Series Models
Models
Trend
Seasonality
Holiday Effects
Cycle
Suitable for
Signal Types
ARIMA
Yes
(learned by linear 
regression)
Yes
(limited season 
length)
No
No
—
Convolution
Yes
(learned by CNNs)
Yes
(limited window size)
Yes
(learned by CNNs)
Possibly no
(due to limited 
window size)
—
Spectral 
density
Yes
(learned by CNNs)
Yes
(limited window size)
Yes
(learned by CNNs)
Yes
(present in 
spectogram)
Stationary
Wavelet 
analysis
Yes
(learned by CNNs)
Yes
(unlimited window 
size)
Yes
(learned by CNNs)
Yes
(present in 
scalogram)
Stationary and 
non-stationary
Transformer 
models
Yes
(learned by attention)
Yes
(limited time delay)
Yes
(learned by attention)
Possibly no
(due to limited
time delay)
Stationary and 
non-stationary

---

## Slide 66

Thank You
prachya@siit.tu.ac.th 
kaamanita@gmail.com 

---

