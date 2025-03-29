from sklearn.metrics import confusion_matrix, classification_report

# Predict on the validation set
y_pred = model.predict(X_val)
y_pred_classes = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_classes)
print(cm)

# Classification Report
cr = classification_report(y_val, y_pred_classes)
print(cr)
