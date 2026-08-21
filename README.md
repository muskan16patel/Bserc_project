# Perimeter Threat Assessment and Decision Support System

A multi-sensor **Edge AI-based perimeter security system** designed to detect, classify, localize, and track potential threats while reducing false alarms and operating under limited network connectivity.

## 🚨 Overview

The system combines **acoustic and seismic sensing** with machine learning and rule-based intelligence to identify events occurring around a protected perimeter.

Instead of continuously transmitting raw sensor data, processing is performed at the edge. Each sensor node analyzes its local environment and sends only compact alerts and extracted features to the central fusion layer. This reduces bandwidth requirements, enables offline operation, and prevents raw audio from leaving the sensor node.

## 🏗️ System Architecture

The system consists of three main layers:

### 1. Edge Sensor Layer

* Microphone and ground-vibration (seismic) sensors
* Local signal processing and feature extraction
* Per-node baseline modeling
* Autoencoder-based anomaly detection
* Frequency-band analysis
* Local threat decision-making
* Offline operation

### 2. Fusion & Threat Intelligence Layer

* Acoustic and seismic sensor fusion
* Threat classification
* Severity scoring
* Time Difference of Arrival (TDoA)-based localization
* Movement and direction tracking
* Anti-spoofing checks
* Sensor/node health monitoring
* Confidence estimation

### 3. Command Dashboard

A Streamlit-based dashboard provides:

* Live perimeter map
* Active threat markers
* Threat classification and severity
* Distance and movement direction
* Confidence information
* Sensor/node health status

The architecture is designed so that Layer 1 can continue detecting events even when network connectivity is unavailable, while Layer 2 performs the cross-node processing required for localization and tracking.

## 🧠 Detection Pipeline

```text
Sensor Data
     ↓
Signal Windowing
     ↓
Feature Extraction
     ↓
Autoencoder Anomaly Detection
     +
Frequency-Band Analysis
     ↓
Tiered Fusion
     ↓
Threat Classification
     ↓
Severity Scoring
     ↓
Localization & Tracking
     ↓
Trust / Node Health Assessment
     ↓
Command Dashboard
```

The system classifies detected events into four broad categories:

* 👤 Person
* 🚗 Vehicle
* 🐾 Animal
* 🌧️ Environmental Noise

Environmental and animal events are explicitly modeled so that the system can suppress non-threats and reduce operator alarm fatigue.

## 🤖 Machine Learning Approach

### Autoencoder

A lightweight autoencoder is trained using normal baseline data collected at each sensor location. The reconstruction error is used as an anomaly score.

This allows the system to detect deviations from the normal acoustic/seismic environment without requiring large amounts of labeled intrusion data.

### Frequency Layer

In parallel with the autoencoder, the system analyzes energy across selected frequency bands. This provides an interpretable second detection mechanism and helps identify characteristic patterns associated with footsteps, vehicles, wind, and rain.

### Tiered Fusion

The outputs of the two detection layers are combined into three levels:

```text
Both layers / extreme anomaly → THREAT
One layer fires               → WARNING
Neither layer fires           → CLEAR
```

The dual-layer approach is intended to improve detection while suppressing false alarms.

## 📍 Localization & Tracking

When an event is detected by multiple nodes, **Time Difference of Arrival (TDoA)** information is used to estimate its position.

Successive position estimates allow the system to determine:

* Current location
* Direction of movement
* Whether the event is approaching or retreating
* Estimated movement speed

Reliable timestamps across nodes are therefore an important part of the system design.

## 🛡️ Privacy & Edge Processing

A key design principle is:

> **Raw audio never leaves the sensor node.**

Only the event verdict, timestamp, confidence, and compact extracted features are transmitted.

This design reduces network bandwidth requirements, supports operation over intermittent links, and limits exposure of raw audio data.

## 🧰 Technology Stack

### Software

* **Python**
* **NumPy**
* **SciPy**
* **librosa**
* **TensorFlow / Keras**
* **TensorFlow Lite / ONNX Runtime**
* **scikit-learn**
* **pandas**
* **Streamlit**
* **Folium / PyDeck**
* **Plotly**
* **soundfile**
* **pydub**

### Hardware

* **Raspberry Pi 5** as the primary edge-computing target
* Microphone
* Seismic / ground-vibration sensor

The technology stack was selected to support offline operation on modest edge hardware while reusing components from the project's EdgeForge architecture.

## 📂 Repository Structure

```text
perimeter-threat-system/
│
├── src/
│   ├── sensors/
│   ├── preprocessing/
│   ├── features/
│   ├── anomaly_detection/
│   ├── classification/
│   ├── fusion/
│   ├── localization/
│   ├── tracking/
│   └── trust/
│
├── dashboard/
│   ├── app.py
│   └── cache/
│
├── notebooks/
│
├── data/
│   ├── real/
│   └── synthetic/
│
├── results/
│
├── tests/
│
├── docs/
│
├── requirements.txt
└── README.md
```

The repository separates source code, synthetic data, dashboard components, notebooks, results, tests, and documentation to improve reproducibility and maintainability.

## 🎯 Project Objectives

The system is evaluated on:

* Threat classification performance
* Suppression of animal/environmental false alarms
* Localization accuracy
* Confidence degradation under node failure
* Demonstration reliability

The project specifically targets **zero false alarms in animal/environmental scenarios** and measures classification and localization independently rather than relying on a single overall accuracy number.

## 🚀 Demonstration Modes

The system supports three demonstration modes:

1. **Live Simulation** – full detection pipeline running on scenario data
2. **Scenario Playback** – pre-computed inference results replayed through the dashboard
3. **Fallback Mode** – cached results used when computation or hardware fails

All three modes are designed to produce the same dashboard output, improving demonstration reliability.

## 🔮 Future Work

Potential extensions include:

* Federated learning across geographically separated command posts
* Acoustic propagation correction based on terrain and temperature
* Automatic camera integration and slewing
* Deployment on production-grade edge hardware
* Improved threat classification
* Larger real-world datasets and field validation

These extensions are intentionally kept outside the current implementation scope to maintain a manageable and demonstrable system.

## 👥 Team

**Jaypee Institute of Information Technology, Noida**

* Alok Srivastava
* Kritika Arora
* Utkarsh Srivastava
* Muskan patel

**BSERC Internship — Defence and Perimeter Security**

---

## 📌 Disclaimer

This project is an academic/internship prototype intended for research, experimentation, and demonstration. Performance in real-world deployment depends on sensor placement, environmental conditions, network availability, calibration, and field validation.
