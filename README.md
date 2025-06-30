# 🥔 Potato Leaf Disease Detection using Deep Learning

A web application to detect **Potato Leaf Diseases** (Early Blight, Late Blight, Healthy) using a Convolutional Neural Network (CNN) model. It also visualizes model attention using **Grad-CAM Heatmaps**.

Built with **TensorFlow**, **Streamlit**, and **OpenCV**.

---

![image](https://github.com/user-attachments/assets/ecd007ea-2e57-4ae5-b2c7-36f890f6fa68)

![image](https://github.com/user-attachments/assets/d599718f-0028-437b-9eb3-78905e69ac1e)

![image](https://github.com/user-attachments/assets/8f861ade-bcbb-443e-ba0d-bbeed9f7cd8d)

![image](https://github.com/user-attachments/assets/3050d848-f5c8-4fa0-b63d-1ee93d78f806)


## 📸 Features

* ✅ Upload potato leaf images
* 🔮 Get real-time predictions for 3 disease classes:

  * Early Blight
  * Late Blight
  * Healthy
* 🔥 Visualize model attention using **Grad-CAM**
* 📊 Bar chart of class probabilities
* 🧠 Model and prediction info in sidebar
* 🥘 View most recent 5 predictions

---

## 🧠 Model Details

* **Architecture:** Custom CNN (4 Conv blocks + FC layers)
* **Input Shape:** 128×128×3
* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam
* **Training Dataset:** 3-class PLD dataset (augmented)

---

## 🧹 Folder Structure

```
Potato-Leaf-Disease-Prediction/
│
├── app/
│   ├── app.py                # Streamlit App
│   ├── model_loader.py       # Model & prediction logic
│   ├── image_utils.py        # Preprocessing helper
│   └── grad_cam.py           # Grad-CAM overlay utils
│
├── model/
│   └── potato_leaf_model.h5  # 🔗 Downloaded separately (see below)
│
├── requirements.txt
└── README.md
```

---

<details>
<summary>⬇️ Model Download</summary>

The trained `.h5` model is hosted on Hugging Face. Download it manually and place it in the `model/` folder.

**📅 Download:**
[https://huggingface.co/venkatram-2005/Potato-Leaf-Disease/raw/main/potato\_leaf\_model.h5](https://huggingface.co/venkatram-2005/Potato-Leaf-Disease/raw/main/potato_leaf_model.h5)

**📁 Save as:**

```
model/potato_leaf_model.h5
```

</details>

---

<details>
<summary>▶️ How to Run Locally</summary>

```bash
# 1. Clone the repository
$ git clone https://github.com/venkatram-2005/Potato-Leaf-Disease-Prediction.git
$ cd Potato-Leaf-Disease-Prediction

# 2. Create and activate virtual environment
$ python -m venv venv
$ venv\Scripts\activate     # On Windows
# OR
$ source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Run the app
$ streamlit run app/app.py
```

</details>

---

<details>
<summary>📜 License (MIT)</summary>

```
MIT License

Copyright (c) 2025 Valluri Venkatram

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

</details>

---

## 🙌 Acknowledgments

* TensorFlow
* Streamlit
* Grad-CAM
* [Hugging Face](https://huggingface.co/) for free model hosting

> 💡 *This project showcases explainable deep learning for plant disease detection.*
