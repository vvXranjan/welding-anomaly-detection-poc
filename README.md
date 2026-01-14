# Welding Anomaly Detection – PatchCore PoC

This repository contains an industry-style Proof of Concept (PoC) for
automated weld inspection using unsupervised anomaly detection.

## Problem Statement
Manual weld inspection is time-consuming and inconsistent. Labeled defect
data is limited, making supervised models difficult to deploy.

## Solution Overview
- Train PatchCore on **normal weld images only**
- Detect deviations as anomalies
- Generate **anomaly scores** and **explainable heatmaps**

## Pipeline
Weld Image → Preprocessing → PatchCore → Anomaly Score + Heatmap → Decision

## Tech Stack
- PyTorch
- Anomalib (PatchCore)
- OpenCV
- NumPy

## Current Status
✅ Model trained  
✅ Inference working  
✅ Thresholding validated  
✅ Demo-ready outputs  

## Future Scope
- Real-time camera integration
- MLOps deployment
- Dashboard & BI integration

