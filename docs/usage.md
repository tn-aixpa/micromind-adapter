# MicroMind Usage Guide

## Training Models

MicroMind provides a flexible framework for training neural networks optimized for low resources. Here's how to use it:

### Basic Training Example

```python
import micromind as mm
from micromind.networks import PhiNet, XiNet
from micromind.utils import parse_configuration

# 1. Define your model class
class ImageClassification(mm.MicroMind):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        
        if hparams.model == "phinet":
            self.modules["classifier"] = PhiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_layers=hparams.num_layers,
                num_classes=hparams.num_classes,
                include_top=True
            )
        elif hparams.model == "xinet":
            self.modules["classifier"] = XiNet(
                input_shape=hparams.input_shape,
                alpha=hparams.alpha,
                num_classes=hparams.num_classes,
                include_top=True
            )

# 2. Setup training
hparams = parse_configuration("path/to/config.py")
train_loader, val_loader = create_loaders(hparams)

# 3. Create experiment folder and checkpointer
exp_folder = mm.utils.checkpointer.create_experiment_folder(
    hparams.output_folder, 
    hparams.experiment_name
)
checkpointer = mm.utils.checkpointer.Checkpointer(
    exp_folder, 
    hparams=hparams, 
    key="loss"
)

# 4. Initialize model and metrics
model = ImageClassification(hparams=hparams)
top1 = mm.Metric("top1_acc", top_k_accuracy(k=1), eval_only=True)
top5 = mm.Metric("top5_acc", top_k_accuracy(k=5), eval_only=True)

# 5. Train the model
model.train(
    epochs=hparams.epochs,
    datasets={"train": train_loader, "val": val_loader},
    metrics=[top5, top1],
    checkpointer=checkpointer
)
```

### Configuration

Models can be configured using configuration files. Example configuration for PhiNet:

```python
# config.py
model = "phinet"
input_shape = (32, 32, 3)
alpha = 1.0
num_layers = 5
num_classes = 10
epochs = 100
```

## Available Models

MicroMind provides several model architectures optimized for microcontrollers:

1. **PhiNet**: Efficient architecture with configurable parameters
   - `alpha`: Network width multiplier
   - `num_layers`: Depth of the network
   - `input_shape`: Input dimensions
   
2. **XiNet**: Alternative architecture with different trade-offs
   - `alpha`: Network width multiplier
   - `gamma`: Additional scaling parameter
   - `input_shape`: Input dimensions

## Model Checkpointing

MicroMind automatically handles model checkpointing:

```python
# Get the latest checkpoint
save_path = "path/to/experiment/save"
latest_checkpoint = get_latest_checkpoint(save_path)
```

## Monitoring Training

The framework provides built-in metrics tracking:
- Top-1 Accuracy
- Top-5 Accuracy
- Loss monitoring
- Parameter count
- MAC operations

## Best Practices

1. **Model Selection**
   - Use PhiNet for balanced performance/size
   - Use XiNet when targeting specific constraints

2. **Training Configuration**
   - Start with default hyperparameters
   - Adjust alpha and num_layers based on target device
   - Use appropriate input_shape for your data

3. **Performance Optimization**
   - Monitor training metrics
   - Use checkpointing for long training runs
   - Validate on target hardware

For more examples and detailed API documentation, visit the [MicroMind GitHub repository](https://github.com/micromind-toolkit/micromind).
