#!/usr/bin/env python3
"""
Fire Fingerprint CNN Model
Multi-task CNN architecture for fire pattern classification
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path

class FireCNN:
    """Multi-task CNN for fire fingerprint classification"""
    
    def __init__(self, input_shape=(224, 224, 4), num_classes=None):
        self.input_shape = input_shape
        self.num_classes = num_classes or {
            'fire_type': 3,
            'ignition_cause': 11,  # 10 + Other
            'state': 8,
            'size_category': 4
        }
        self.model = None
        self.history = None
    
    def build_custom_cnn(self):
        """Build custom CNN architecture optimized for fire fingerprints"""
        inputs = layers.Input(shape=self.input_shape, name='fingerprint_input')
        
        # Initial convolution to handle 4-channel input
        x = layers.Conv2D(32, (7, 7), strides=2, padding='same', name='initial_conv')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.ReLU(name='initial_relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='initial_pool')(x)
        
        # Convolutional blocks
        # Block 1
        x = self._conv_block(x, 64, 'block1')
        x = self._conv_block(x, 64, 'block1_2')
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        
        # Block 2
        x = self._conv_block(x, 128, 'block2')
        x = self._conv_block(x, 128, 'block2_2')
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        
        # Block 3
        x = self._conv_block(x, 256, 'block3')
        x = self._conv_block(x, 256, 'block3_2')
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        
        # Block 4
        x = self._conv_block(x, 512, 'block4')
        x = self._conv_block(x, 512, 'block4_2')
        
        # Global pooling and feature extraction
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        # Shared dense layers
        x = layers.Dense(1024, activation='relu', name='shared_dense1')(x)
        x = layers.BatchNormalization(name='shared_bn1')(x)
        x = layers.Dropout(0.5, name='dropout2')(x)
        
        shared_features = layers.Dense(512, activation='relu', name='shared_features')(x)
        
        # Task-specific heads
        outputs = {}
        
        # Fire type classification
        fire_type_branch = layers.Dense(256, activation='relu', name='fire_type_dense')(shared_features)
        fire_type_branch = layers.Dropout(0.3, name='fire_type_dropout')(fire_type_branch)
        outputs['fire_type'] = layers.Dense(
            self.num_classes['fire_type'], 
            activation='softmax', 
            name='fire_type'
        )(fire_type_branch)
        
        # Ignition cause classification
        cause_branch = layers.Dense(256, activation='relu', name='cause_dense')(shared_features)
        cause_branch = layers.Dropout(0.3, name='cause_dropout')(cause_branch)
        outputs['ignition_cause'] = layers.Dense(
            self.num_classes['ignition_cause'], 
            activation='softmax', 
            name='ignition_cause'
        )(cause_branch)
        
        # State classification
        state_branch = layers.Dense(128, activation='relu', name='state_dense')(shared_features)
        state_branch = layers.Dropout(0.3, name='state_dropout')(state_branch)
        outputs['state'] = layers.Dense(
            self.num_classes['state'], 
            activation='softmax', 
            name='state'
        )(state_branch)
        
        # Size category classification
        size_branch = layers.Dense(128, activation='relu', name='size_dense')(shared_features)
        size_branch = layers.Dropout(0.3, name='size_dropout')(size_branch)
        outputs['size_category'] = layers.Dense(
            self.num_classes['size_category'], 
            activation='softmax', 
            name='size_category'
        )(size_branch)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='FireCNN')
        
        return model
    
    def _conv_block(self, x, filters, block_name):
        """Convolutional block with batch normalization and ReLU"""
        x = layers.Conv2D(filters, (3, 3), padding='same', name=f'{block_name}_conv')(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn')(x)
        x = layers.ReLU(name=f'{block_name}_relu')(x)
        return x
    
    def build_transfer_learning_model(self, base_model='efficientnet'):
        """Build model using transfer learning from pre-trained networks"""
        
        # Handle 4-channel input by creating adapter layer
        inputs = layers.Input(shape=self.input_shape, name='fingerprint_input')
        
        # Convert 4-channel to 3-channel for pre-trained models
        # Use learned combination of channels
        channel_adapter = layers.Conv2D(3, (1, 1), padding='same', name='channel_adapter')(inputs)
        
        # Base model
        if base_model == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_tensor=channel_adapter
            )
        elif base_model == 'resnet':
            base = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_tensor=channel_adapter
            )
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Freeze base model initially
        base.trainable = False
        
        # Add custom head
        x = base.output
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        
        # Shared features
        shared_features = layers.Dense(512, activation='relu', name='shared_features')(x)
        
        # Task-specific outputs (same as custom model)
        outputs = self._create_task_heads(shared_features)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=f'Fire_{base_model}')
        
        return model
    
    def _create_task_heads(self, shared_features):
        """Create task-specific classification heads"""
        outputs = {}
        
        # Fire type
        fire_type_branch = layers.Dense(128, activation='relu')(shared_features)
        fire_type_branch = layers.Dropout(0.3)(fire_type_branch)
        outputs['fire_type'] = layers.Dense(
            self.num_classes['fire_type'], 
            activation='softmax', 
            name='fire_type'
        )(fire_type_branch)
        
        # Ignition cause
        cause_branch = layers.Dense(128, activation='relu')(shared_features)
        cause_branch = layers.Dropout(0.3)(cause_branch)
        outputs['ignition_cause'] = layers.Dense(
            self.num_classes['ignition_cause'], 
            activation='softmax', 
            name='ignition_cause'
        )(cause_branch)
        
        # State
        state_branch = layers.Dense(64, activation='relu')(shared_features)
        state_branch = layers.Dropout(0.3)(state_branch)
        outputs['state'] = layers.Dense(
            self.num_classes['state'], 
            activation='softmax', 
            name='state'
        )(state_branch)
        
        # Size category
        size_branch = layers.Dense(64, activation='relu')(shared_features)
        size_branch = layers.Dropout(0.3)(size_branch)
        outputs['size_category'] = layers.Dense(
            self.num_classes['size_category'], 
            activation='softmax', 
            name='size_category'
        )(size_branch)
        
        return outputs
    
    def compile_model(self, learning_rate=0.001, loss_weights=None):
        """Compile the model with appropriate losses and metrics"""
        
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Default loss weights (can be tuned based on task importance)
        if loss_weights is None:
            loss_weights = {
                'fire_type': 1.0,
                'ignition_cause': 1.0,
                'state': 0.5,  # Less important for fire science
                'size_category': 1.5  # More important for fire management
            }
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss={
                'fire_type': 'sparse_categorical_crossentropy',
                'ignition_cause': 'sparse_categorical_crossentropy',
                'state': 'sparse_categorical_crossentropy',
                'size_category': 'sparse_categorical_crossentropy'
            },
            loss_weights=loss_weights,
            metrics={
                'fire_type': ['accuracy'],
                'ignition_cause': ['accuracy'],
                'state': ['accuracy'],
                'size_category': ['accuracy']
            }
        )
        
        print("Model compiled successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
    def create_callbacks(self, model_save_path='best_fire_model.h5', patience=10):
        """Create training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger('training_log.csv')
        ]
        
        return callbacks_list
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        
        if self.model is None:
            raise ValueError("Model not compiled yet. Call compile_model() first.")
        
        # Prepare validation data
        if X_val is None and y_val is None:
            validation_data = None
            validation_split = validation_split
        else:
            validation_data = (X_val, y_val)
            validation_split = None
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(X_train):,}")
        print(f"Batch size: {batch_size}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        print("Training completed!")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test, class_names=None):
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        print("Evaluating model...")
        
        # Get predictions
        predictions = self.model.predict(X_test, verbose=1)
        
        # Evaluate each task
        results = {}
        
        for task in ['fire_type', 'ignition_cause', 'state', 'size_category']:
            y_true = y_test[task]
            y_pred = np.argmax(predictions[task], axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred)
            results[task] = {'accuracy': accuracy}
            
            print(f"\n{task.upper()} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            if class_names and task in class_names:
                report = classification_report(
                    y_true, y_pred, 
                    target_names=class_names[task],
                    output_dict=True
                )
                results[task]['classification_report'] = report
                print("Classification Report:")
                print(classification_report(y_true, y_pred, target_names=class_names[task]))
        
        return results, predictions
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Overall Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Task-specific accuracies
        tasks = ['fire_type', 'ignition_cause', 'size_category']
        colors = ['blue', 'red', 'green']
        
        for i, (task, color) in enumerate(zip(tasks, colors)):
            if f'{task}_accuracy' in self.history.history:
                axes[0, 1].plot(
                    self.history.history[f'{task}_accuracy'], 
                    label=f'{task} (train)', 
                    color=color, 
                    linestyle='-'
                )
                if f'val_{task}_accuracy' in self.history.history:
                    axes[0, 1].plot(
                        self.history.history[f'val_{task}_accuracy'], 
                        label=f'{task} (val)', 
                        color=color, 
                        linestyle='--'
                    )
        
        axes[0, 1].set_title('Task Accuracies')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Task-specific losses
        for i, (task, color) in enumerate(zip(tasks, colors)):
            if f'{task}_loss' in self.history.history:
                axes[1, 1].plot(
                    self.history.history[f'{task}_loss'], 
                    label=f'{task}', 
                    color=color
                )
        
        axes[1, 1].set_title('Task-Specific Losses')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_feature_extractor(self):
        """Get feature extraction model (without classification heads)"""
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Extract features from shared_features layer
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('shared_features').output
        )
        
        return feature_extractor

def create_fire_cnn(architecture='custom', input_shape=(224, 224, 4), num_classes=None):
    """Factory function to create FireCNN model"""
    
    fire_cnn = FireCNN(input_shape=input_shape, num_classes=num_classes)
    
    if architecture == 'custom':
        fire_cnn.model = fire_cnn.build_custom_cnn()
    elif architecture in ['efficientnet', 'resnet']:
        fire_cnn.model = fire_cnn.build_transfer_learning_model(architecture)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return fire_cnn

# Test function
def test_model():
    """Test model creation and compilation"""
    print("Testing FireCNN model...")
    
    # Create model
    fire_cnn = create_fire_cnn(architecture='custom')
    
    # Compile model
    fire_cnn.compile_model()
    
    # Print model summary
    fire_cnn.model.summary()
    
    # Test with dummy data
    dummy_input = np.random.random((10, 224, 224, 4))
    dummy_labels = {
        'fire_type': np.random.randint(0, 3, 10),
        'ignition_cause': np.random.randint(0, 11, 10),
        'state': np.random.randint(0, 8, 10),
        'size_category': np.random.randint(0, 4, 10)
    }
    
    # Test prediction
    predictions = fire_cnn.model.predict(dummy_input)
    print(f"Prediction shapes:")
    for task, pred in predictions.items():
        print(f"  {task}: {pred.shape}")
    
    print("Model test successful!")

if __name__ == "__main__":
    test_model()
