# Architecture

## CNN

```mermaid
flowchart TD
    A[Input: 32x32x3] --> B[Conv2D: 32 filters, 3x3, ReLU]
    B --> C[MaxPooling: 2x2]
    C --> D[Conv2D: 64 filters, 3x3, ReLU]
    D --> E[MaxPooling: 2x2]
    E --> F[Conv2D: 64 filters, 3x3, ReLU]
    F --> G[Flatten]
    G --> H[Dense: 64, ReLU]
    H --> I[Dense: 10, Softmax]
    I --> J[Predicted CIFAR-10 class]
```

## Layer Roles

- `Input`: receives 32×32 RGB images.
- `Conv2D`: extracts spatial features.
- `MaxPooling2D`: reduces spatial dimensions.
- `Flatten`: converts feature maps into a vector.
- `Dense(64)`: learns a classification representation.
- `Dense(10)`: outputs probabilities for the ten classes.

The layer sequence and parameters are taken from the supplied guide.
