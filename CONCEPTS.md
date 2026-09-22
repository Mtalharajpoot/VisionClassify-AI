# Core Concepts

## Conv2D

A convolution layer learns spatial patterns. The supplied guide describes early features such as:

- edges
- corners
- lines
- textures

Deeper convolution layers can learn more complex structures.

## Kernel / Filter

A `(3,3)` kernel is a small moving window over the image.

## ReLU

`activation="relu"` introduces non-linearity.

## Max Pooling

`MaxPooling2D((2,2))` reduces spatial dimensions while retaining strong features.

## Dense Layer

Dense layers use the learned features for classification.

## Softmax

`Dense(10, activation="softmax")` produces probabilities across the 10 classes.

## Optimizer

Adam updates model weights during training.

## Loss

`sparse_categorical_crossentropy` measures how wrong the classification is for the integer class labels.

## Accuracy

Accuracy is the proportion of correctly classified samples.

## Epoch

One complete pass through the training data.

## Batch Size

The guide uses 64 images per training batch.

## Validation Split

`validation_split=0.2` reserves approximately 20% of the original training set for validation.

## Overfitting

A pattern such as increasing training accuracy while validation accuracy decreases can indicate overfitting.

## Training / Validation / Test

```text
Training   -> model learns
Validation -> development monitoring
Test       -> final independent evaluation
```
