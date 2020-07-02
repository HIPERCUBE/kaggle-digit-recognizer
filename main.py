import numpy as np
import pandas as pd
import tensorflow as tf

import model

# Load train data
train = pd.read_csv('data/train.csv')
train_x = train.drop('label', axis=1).values / 255.0
train_y = train['label'].values

# Load test data
test = pd.read_csv('data/test.csv')
test_x = test.values / 255.0

# Convert flatten x data into 2d array
train_x = train_x.reshape(train_x.shape[0], 28, 28)
test_x = test_x.reshape(test_x.shape[0], 28, 28)

# Load model
model = model.build_model()

# Train model
model.fit(train_x, train_y, epochs=10)

# Predict test data and create submission file
preds = pd.DataFrame({
    'ImageId': list(range(1, test.shape[0] + 1)),
    'Label': model.predict_classes(test_x)
})
preds.to_csv('submission.csv', index=False, header=True)
