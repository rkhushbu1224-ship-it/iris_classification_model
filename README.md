# 🌸 Iris Classifier – Flask Web App

A full-stack machine learning web application that predicts the species of an iris flower using a Gaussian Naive Bayes model.

---

## 🚀 Features

* Predict iris species (Setosa, Versicolor, Virginica)
* Interactive sliders for feature input
* Real-time predictions using Flask API
* Model accuracy & cross-validation stats
* Confusion matrix visualization
* Classification report
* Prediction history tracking
* Beautiful responsive UI

---

## 🧠 Machine Learning Model

* Algorithm: Gaussian Naive Bayes
* Dataset: Iris Dataset
* Library: scikit-learn

---

## 📊 Model Performance

* Test Accuracy: ~{{accuracy}}
* Cross-validation Accuracy: ~{{cv}}

---

## 🛠️ Tech Stack

* Backend: Python, Flask
* ML: scikit-learn, NumPy, Pandas
* Frontend: HTML, CSS, JavaScript
* Model Storage: joblib

---

## 📂 Project Structure

```
iris_classification_model/
│
├── iris_app.py          # Main Flask application
├── iris_classifier_nb.joblib   # Saved ML model
├── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/iris_classification_model.git
cd iris_classification_model
```

### 2️⃣ Install dependencies

```
pip install flask scikit-learn numpy pandas joblib
```

### 3️⃣ Run the application

```
python iris_app.py
```

### 4️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 📸 Screenshots

(Add screenshots of your UI here)

---

## 💡 How it works

1. User inputs flower measurements using sliders
2. Data is sent to Flask backend
3. Model predicts species using Naive Bayes
4. Results are displayed with probabilities

---

## 📌 Future Improvements

* Deploy on cloud (Render / Heroku)
* Add more ML models (SVM, KNN)
* Improve UI animations
* Add dataset upload feature

---

## 👩‍💻 Author

Khushbu R

---

## ⭐ Show your support

If you like this project, give it a ⭐ on GitHub!
