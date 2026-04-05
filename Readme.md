[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nSn4fJNC)

# Raqeeb — Stolen License Plate Detection & Vehicle Verification System

##  Project Overview

**Raqeeb (رقيب)** is an intelligent Computer Vision system designed to detect stolen license plates and verify whether a plate belongs to the correct vehicle.

The system analyzes an input image, detects the vehicle, extracts and reads the license plate using a Vision-Language Model, predicts the car brand and model, and compares the results with a database.

This project goes beyond traditional plate recognition by adding a **smart verification layer** to detect suspicious or mismatched vehicles.

---

## 🎯 Project Objectives

* Detect vehicles from images
* Extract and read license plates
* Predict vehicle brand and model
* Compare results with a database
* Detect stolen plates
* Detect mismatched vehicles (fake plate usage)

This system is important for:

* Traffic security 
* Law enforcement 
* Smart city systems 
* Preventing fraud and illegal vehicle usage 

---

## 🧠 Methodology

The system follows a complete pipeline:

1. **Vehicle Detection**
   Detect the vehicle using a deep learning model (YOLO)

2. **Vehicle Cropping**
   Extract the vehicle region from the image

3. **Vehicle Analysis (Brand & Model Prediction)**
   Predict:

   * Car brand
   * Car model

4. **Plate Detection**
   Detect the license plate area

5. **Plate Enhancement**
   Improve image quality

6. **Plate Reading using Vision-Language Model**
   The system uses Qwen to extract the license plate number instead of traditional OCR.

7. **Database Matching**
   Retrieve stored data using the plate number:

   * Expected brand
   * Expected model
   * Stolen status

8. **Smart Verification (Core Innovation)**
   The system compares:

   * Detected vehicle
   * Database record

   Then decides:

   * **Stolen Plate** → if plate is flagged
   * **Valid Vehicle** → if everything matches

---

## 📂 Dataset

The project uses:

* Public vehicle image datasets
* Custom test images
* Simulated database of license plates

### Preprocessing

* Image resizing
* Object cropping
* Plate enhancement for better recognition

---

## ⚙️ Implementation

### Tools & Technologies

* Python
* OpenCV
* YOLO (Object Detection)
* Vehicle Classification Model (Brand / Model)
* Qwen (Plate Reading)
*  Database

### Key Features

* End-to-end AI pipeline
* Plate recognition using Vision-Language Model
* Vehicle brand/model prediction
* Smart verification system

---

## 🧠 Key Innovation

Unlike traditional systems, **Raqeeb** does not only read license plates.

It verifies whether the plate actually belongs to the detected vehicle by comparing it with stored data.

This enables detection of:

* Fake plates
* Plate swapping
* Suspicious vehicles

---

## 📊 Results

The system successfully:

* Detects vehicles in images
* Reads license plates using Qwen
* Predicts vehicle brand and model
* Detects mismatches and stolen plates

### Example Output

Plate Number: XYZ1234
Detected Car: Toyota Camry
Database Record: Hyundai Elantra

Status: ⚠️ MISMATCH DETECTED

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
python main.py
```

---

## 📁 Project Structure

```bash
project/
│── data/
│── models/
│── utils/
│── test_images/
│── outputs/
│── main.py
│── requirements.txt
│── README.md
```

---

## 👥 Team Members

* Abdulrahman Almutairi
* Mooj Algoot
* Majid Alnodali
* Lama Aldaham

---

## 🔮 Future Improvements

* Real-time camera integration
* Connection to official government databases
* Improved vehicle model accuracy
* Confidence scoring system


---

## ✅ Conclusion

**Raqeeb** is a smart AI-powered system that combines Computer Vision and deep learning to detect stolen license plates and verify vehicle identity.

It provides a practical and scalable solution for enhancing road security and detecting fraudulent vehicle activity.
