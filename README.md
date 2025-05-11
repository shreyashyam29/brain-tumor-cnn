# Brain Tumor Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify brain MRI images into two categories: Tumor and No Tumor. The model is trained using a dataset of MRI scans and evaluated using common performance metrics and visualizations.

---

##  Dataset

The dataset consists of MRI images categorized as either tumor or no tumor. The data is preprocessed and augmented to improve model performance.

---

##  Model Architecture

A simple CNN model is used with the following layers:

* Convolutional layers
* MaxPooling layers
* Dropout for regularization
* Flatten + Dense layers with final sigmoid activation

---

##  Evaluation Metrics

* **Accuracy**
* **Precision**: 0.90
* **Recall**: 0.90
* **F1-Score**: 0.90
* **Confusion Matrix**:

  ```
  [[17  3]
   [ 3 28]]
  ```

---

##  Visualizations

### ROC Curve

![ROC Curve](./f8ba90a1-59e7-48f6-8c8b-1ecc561eae07.png)

* AUC Score: **0.88**

### Training History

![Training vs Validation Loss & Accuracy](./51fb235f-45e6-4cd1-9b63-91310083d801.png)

* Shows model improvement across epochs

### Prediction Samples

![True vs Predicted Labels](./89854654-46f9-4537-b2be-eaaffbfa50e0.png)

* Displaying model predictions on test samples

---

## Technologies Used

* Python
* TensorFlow / Keras
* Matplotlib / Seaborn for visualization
* NumPy, Pandas

---

##  How to Run

```bash
pip install -r requirements.txt
python train.py
```

Make sure the dataset is correctly placed in the specified folders before running.

---

## Results Summary

* The model achieves strong generalization performance with balanced precision and recall.
* Visualizations show consistent improvement and minimal overfitting.

---

## Contact

For queries or suggestions, feel free to raise an issue or contact me via GitHub.

---

> **Note**: This project is for educational/research purposes. Model performance may vary with different datasets or configurations.
