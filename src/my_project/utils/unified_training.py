"""
Unified training and inference functions for phase and magnitude models.

This module provides generic training and inference functions that can work
with different model types (PhaseNet, PhaseNetLSTM, PhaseNetMag, AMAG_v2)
while abstracting away the model-specific details.
"""

import torch
import seisbench.data as sbd
from seisbench.data import BenchmarkDataset
from seisbench.models import WaveformModel
from typing import Union, Optional, Dict, Any, Tuple
import logging

# Import specific training functions
from my_project.tutorial.tutorial import train_phasenet, evaluate_phase_model
from my_project.models.phasenetLSTM.train import train_phasenet_lstm_model
from my_project.models.phasenet_mag.train import train_phasenet_mag
from my_project.models.phasenet_mag.evaluate import evaluate_phasenet_mag
from my_project.models.AMAG_v2.train import train_magnitude_net
from my_project.models.AMAG_v2.evaluate import evaluate_magnitude_net
from my_project.models.EQTransformer.train import train_eqtransformer_mag
from my_project.models.EQTransformer.evaluate import evaluate_eqtransformer_mag
from my_project.models.ViT.train import train_vit_magnitude
from my_project.models.ViT.evaluate import evaluate_vit_magnitude

# Import model classes for type checking
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM
from my_project.models.EQTransformer.model import EQTransformerMag
from my_project.models.ViT.model import ViTMagnitudeEstimator
import seisbench.models as sbm


def train_phase_model(
    model: Union[sbm.PhaseNet, PhaseNetLSTM, PhaseNetConvLSTM],
    model_name: str,
    data: BenchmarkDataset,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    **kwargs,
) -> Union[WaveformModel, Tuple[WaveformModel, float]]:
    """
    Unified training function for phase-based models (PhaseNet, PhaseNetLSTM).

    Args:
        model: Phase model instance (PhaseNet, PhaseNetLSTM, or PhaseNetConvLSTM)
        model_name: Name for saving model checkpoints
        data: Dataset to train on
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        **kwargs: Additional model-specific parameters

    Returns:
        Trained model (and best loss for LSTM models)
    """

    # Route to appropriate training function based on model type
    if isinstance(model, (PhaseNetLSTM, PhaseNetConvLSTM)):
        print(f"Training PhaseNet-LSTM model: {model.__class__.__name__}")

        # Ensure model is on the correct device
        model.to_preferred_device(verbose=True)

        # Extract LSTM-specific parameters from kwargs
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden_size = kwargs.get("lstm_hidden_size", None)
        lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        lstm_bidirectional = kwargs.get("lstm_bidirectional", True)
        num_workers = kwargs.get("num_workers", 4)
        early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        return train_phasenet_lstm_model(
            data=data,
            epochs=epochs,
            filter_factor=filter_factor,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_bidirectional=lstm_bidirectional,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_workers=num_workers,
            early_stopping_patience=early_stopping_patience,
            model=model,
            model_name=model_name,
        )
    elif isinstance(model, sbm.PhaseNet):
        print(f"Training standard PhaseNet model")

        # Ensure model is on the correct device
        model.to_preferred_device(verbose=True)

        early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        train_phasenet(
            model_name=model_name,
            model=model,
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
        )
        return model
    else:
        raise ValueError(f"Unsupported phase model type: {type(model)}")


def evaluate_phase_model_unified(
    model: Union[sbm.PhaseNet, PhaseNetLSTM, PhaseNetConvLSTM],
    model_path: str,
    data: BenchmarkDataset,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified evaluation function for phase-based models.

    Args:
        model: Phase model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        **kwargs: Additional evaluation parameters

    Returns:
        Dictionary containing evaluation results
    """
    # Ensure model is on the correct device
    model.to_preferred_device(verbose=True)

    print(f"Evaluating phase model: {model.__class__.__name__}")

    # All phase models can use the same evaluation function from tutorial
    results = evaluate_phase_model(model=model, model_path=model_path, data=data)

    return {"model_type": "phase", "results": results}


def train_magnitude_model(
    model: Union[PhaseNetMag, MagnitudeNet, EQTransformerMag, ViTMagnitudeEstimator],
    model_name: str,
    data: BenchmarkDataset,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    optimizer_name: str = "Adam",
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    save_every: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified training function for magnitude-based models (PhaseNetMag, AMAG_v2, EQTransformerMag).

    Args:
        model: Magnitude model instance (PhaseNetMag, MagnitudeNet, EQTransformerMag)
        model_name: Name for saving model checkpoints
        data: Dataset to train on
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        optimizer_name: Optimizer type ("Adam", "AdamW", or "SGD")
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for learning rate scheduler
        save_every: Save model every N epochs
        **kwargs: Additional model-specific parameters

    Returns:
        Dictionary containing training history and results
    """
    # Ensure model is on the correct device
    model.to_preferred_device(verbose=True)

    # Route to appropriate training function based on model type
    if isinstance(model, PhaseNetMag):
        print(f"Training PhaseNetMag model")

        early_stopping_patience = kwargs.get("early_stopping_patience", 10)
        warmup_epochs = kwargs.get("warmup_epochs", 5)

        results = train_phasenet_mag(
            model_name=model_name,
            model=model,
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_patience=scheduler_patience,
            save_every=save_every,
            early_stopping_patience=early_stopping_patience,
            warmup_epochs=warmup_epochs,
        )
        return {"model_type": "phasenet_mag", "results": results}

    elif isinstance(model, MagnitudeNet):
        print(f"Training MagnitudeNet (AMAG_v2) model")

        # Extract AMAG-specific parameters from kwargs
        scheduler_factor = kwargs.get("scheduler_factor", 0.5)
        gradient_clip = kwargs.get("gradient_clip", 1.0)
        warmup_epochs = kwargs.get("warmup_epochs", 5)

        results = train_magnitude_net(
            model_name=model_name,
            model=model,
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            save_every=save_every,
            gradient_clip=gradient_clip,
            warmup_epochs=warmup_epochs,
        )
        return {"model_type": "magnitude_net", "results": results}

    elif isinstance(model, EQTransformerMag):
        print(f"Training EQTransformerMag model")

        # Extract EQTransformerMag-specific parameters from kwargs
        warmup_epochs = kwargs.get("warmup_epochs", 5)
        scheduler_factor = kwargs.get("scheduler_factor", 0.5)
        gradient_clip = kwargs.get("gradient_clip", 1.0)
        early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        results = train_eqtransformer_mag(
            model_name=model_name,
            model=model,
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            save_every=save_every,
            gradient_clip=gradient_clip,
            early_stopping_patience=early_stopping_patience,
            warmup_epochs=warmup_epochs,
        )
        return {"model_type": "eqtransformer_mag", "results": results}

    elif isinstance(model, ViTMagnitudeEstimator):
        print(f"Training ViT Magnitude Estimator model")

        # Extract ViT-specific parameters from kwargs
        gradient_clip = kwargs.get("gradient_clip", 1.0)
        early_stopping_patience = kwargs.get("early_stopping_patience", 20)
        warmup_epochs = kwargs.get("warmup_epochs", 10)
        scheduler_factor = kwargs.get("scheduler_factor", 0.5)

        # Adjust defaults for ViT if not explicitly provided
        if learning_rate == 1e-3:  # Default learning rate, adjust for ViT
            learning_rate = 1e-4
        if batch_size == 256:  # Default batch size, adjust for ViT
            batch_size = 64
        if weight_decay == 1e-5:  # Default weight decay, adjust for ViT
            weight_decay = 1e-2

        results = train_vit_magnitude(
            model_name=model_name,
            model=model,
            data=data,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            save_every=save_every,
            gradient_clip=gradient_clip,
            early_stopping_patience=early_stopping_patience,
            warmup_epochs=warmup_epochs,
        )
        return {"model_type": "vit_magnitude", "results": results}

    else:
        raise ValueError(f"Unsupported magnitude model type: {type(model)}")


def evaluate_magnitude_model(
    model: Union[PhaseNetMag, MagnitudeNet, EQTransformerMag, ViTMagnitudeEstimator],
    model_path: str,
    data: BenchmarkDataset,
    batch_size: int = 256,
    plot_examples: bool = False,
    num_examples: int = 5,
) -> Dict[str, Any]:
    """
    Unified evaluation function for magnitude-based models.

    Args:
        model: Magnitude model instance (PhaseNetMag, MagnitudeNet, EQTransformerMag)
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        **kwargs: Additional evaluation parameters

    Returns:
        Dictionary containing evaluation results
    """
    # Ensure model is on the correct device
    model.to_preferred_device(verbose=True)

    # Route to appropriate evaluation function based on model type
    if isinstance(model, PhaseNetMag):
        print(f"Evaluating PhaseNetMag model")

        results = evaluate_phasenet_mag(
            model=model,
            model_path=model_path,
            data=data,
            batch_size=batch_size,
            plot_examples=plot_examples,
            num_examples=num_examples,
        )
        return {"model_type": "phasenet_mag", "results": results}

    elif isinstance(model, MagnitudeNet):
        print(f"Evaluating MagnitudeNet (AMAG_v2) model")

        results = evaluate_magnitude_net(
            model=model,
            model_path=model_path,
            data=data,
            batch_size=batch_size,
            plot_examples=plot_examples,
            num_examples=num_examples,
        )
        return {"model_type": "magnitude_net", "results": results}

    elif isinstance(model, EQTransformerMag):
        print(f"Evaluating EQTransformerMag model")

        results = evaluate_eqtransformer_mag(
            model=model,
            model_path=model_path,
            data=data,
            batch_size=batch_size,
            plot_examples=plot_examples,
            num_examples=num_examples,
        )
        return {"model_type": "eqtransformer_mag", "results": results}

    elif isinstance(model, ViTMagnitudeEstimator):
        print(f"Evaluating ViT Magnitude Estimator model")

        # Adjust batch size for ViT if using default
        if batch_size == 256:
            batch_size = 64

        results = evaluate_vit_magnitude(
            model=model,
            model_path=model_path,
            data=data,
            batch_size=batch_size,
            plot_examples=plot_examples,
            num_examples=num_examples,
        )
        return {"model_type": "vit_magnitude", "results": results}

    else:
        raise ValueError(f"Unsupported magnitude model type: {type(model)}")
