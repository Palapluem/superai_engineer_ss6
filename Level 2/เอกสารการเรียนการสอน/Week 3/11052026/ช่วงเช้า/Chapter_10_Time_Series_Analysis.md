# Chapter 10: Time Series Analysis

**Prachya Boonkwan (Arm)**  
School of ICT, SIIT, Thammasat University  
prachya@siit.tu.ac.th, kaamanita@gmail.com  

---

## 2. Resources

Slide Material: [https://tinyurl.com/5b6ch5px](https://tinyurl.com/5b6ch5px)

---

## 3. Who? Me?

- **Nickname:** Arm (P'/N' Arm, etc.)
- **Born:** Aug 1981
- **Work:**
  - Researcher at NECTEC 2005-2024
  - Lecturer at SIIT, Thammasat University 2025-now
- **Education:**
  - B.Eng & M.Eng, CPE Kasetsart University
  - Obtained Ministry of Science Scholarship in early 2008
  - Did a PhD in Informatics (AI & Computational Linguistics) at University of Edinburgh, UK from 2008 to 2013 (4.5 years)

---

## 4. Outline

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

# 1. Introduction

---

## 6. Time Series

- A **sequence** of data points read at **equally spaced points of time**
- *E.g.* daily stock price, monthly rice price, heights of ocean tides, audio signals, counts of celestial meteorites, and activity of tectonic plates
- **Time series forecasting**
  - Predicting future values based on previously observed values
- **Stochastic process:** observations close together in time are more closely related than those further apart

---

## 7. E-Commerce Purchase Amounts

![E-Commerce Purchase Amounts chart showing daily sums over time](images/slide_7_img_1.png)

*A visualization showing daily purchase amounts with observable upward trends and cyclical behavior.*

---