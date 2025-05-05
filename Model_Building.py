

# Input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_dim=X_resampled.shape[1]))
model.add(Dropout(0.5))

# Second hidden layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_resampled, y_resampled, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])
