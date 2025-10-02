#!/usr/bin/env python3
"""
Fire CNN Training Pipeline
Complete training system for fire fingerprint classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from cnn_model import create_fire_cnn
from data_processor import FireDataProcessor

class FireTrainer:
    """Complete training pipeline for fire fingerprint CNN"""
    
    def __init__(self, data_dir="processed_data", model_dir="models", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.fire_cnn = None
        self.encoders = None
        self.class_weights = None
        self.training_stats = {}
    
    def load_processed_data(self):
        """Load processed fingerprint data"""
        print("Loading processed data...")
        
        processor = FireDataProcessor("", output_dir=self.data_dir)
        fingerprints, labels, metadata, encoders = processor.load_processed_data()
        
        if fingerprints is None:
            raise ValueError("No processed data found. Run data processing first.")
        
        self.encoders = encoders
        
        print(f"Loaded {len(fingerprints):,} fingerprints")
        print(f"Fingerprint shape: {fingerprints.shape}")
        
        return fingerprints, labels, metadata
    
    def prepare_training_data(self, fingerprints, labels, test_size=0.2, val_size=0.2, random_state=42):
        """Prepare training, validation, and test sets"""
        print("Preparing training data...")
        
        # Convert labels to arrays
        y_fire_type = np.array([l['fire_type'] for l in labels])
        y_ignition_cause = np.array([l['ignition_cause'] for l in labels])
        y_state = np.array([l['state'] for l in labels])
        y_size_category = np.array([l['size_category'] for l in labels])
        
        # Combine labels for stratified split
        y_combined = {
            'fire_type': y_fire_type,
            'ignition_cause': y_ignition_cause,
            'state': y_state,
            'size_category': y_size_category
        }
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            fingerprints, y_combined, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_fire_type  # Stratify by main task
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp['fire_type']
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        # Calculate class distributions
        self._analyze_class_distributions(y_train, y_val, y_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _analyze_class_distributions(self, y_train, y_val, y_test):
        """Analyze class distributions across splits"""
        print("\nClass distributions:")
        
        for task in ['fire_type', 'ignition_cause', 'state', 'size_category']:
            print(f"\n{task.upper()}:")
            
            # Training distribution
            train_dist = pd.Series(y_train[task]).value_counts().sort_index()
            val_dist = pd.Series(y_val[task]).value_counts().sort_index()
            test_dist = pd.Series(y_test[task]).value_counts().sort_index()
            
            print("  Train | Val   | Test  | Class")
            print("  ------|-------|-------|------")
            
            all_classes = sorted(set(y_train[task]) | set(y_val[task]) | set(y_test[task]))
            for cls in all_classes:
                train_count = train_dist.get(cls, 0)
                val_count = val_dist.get(cls, 0)
                test_count = test_dist.get(cls, 0)
                print(f"  {train_count:5d} | {val_count:5d} | {test_count:5d} | {cls}")
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced datasets"""
        print("Calculating class weights...")
        
        self.class_weights = {}
        
        for task in ['fire_type', 'ignition_cause', 'state', 'size_category']:
            classes = np.unique(y_train[task])
            weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=y_train[task]
            )
            
            self.class_weights[task] = dict(zip(classes, weights))
            
            print(f"{task}: {len(classes)} classes, weights range: {weights.min():.2f} - {weights.max():.2f}")
        
        return self.class_weights
    
    def create_model(self, architecture='custom', input_shape=(224, 224, 4)):
        """Create and compile the CNN model"""
        print(f"Creating {architecture} model...")
        
        # Determine number of classes from encoders
        num_classes = {
            'fire_type': len(self.encoders['fire_type']),
            'ignition_cause': len(self.encoders['ignition_cause']),
            'state': len(self.encoders['state']),
            'size_category': len(self.encoders['size_category'])
        }
        
        print(f"Number of classes: {num_classes}")
        
        # Create model
        self.fire_cnn = create_fire_cnn(
            architecture=architecture,
            input_shape=input_shape,
            num_classes=num_classes
        )
        
        # Compile model
        self.fire_cnn.compile_model(learning_rate=0.001)
        
        return self.fire_cnn
    
    def train_model(self, train_data, val_data, epochs=50, batch_size=32, 
                   use_class_weights=True, early_stopping_patience=15):
        """Train the model"""
        
        if self.fire_cnn is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print(f"Starting training...")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Use class weights: {use_class_weights}")
        
        # Prepare class weights for training
        class_weight = None
        if use_class_weights and self.class_weights:
            class_weight = self.class_weights
        
        # Create callbacks
        callbacks = self.fire_cnn.create_callbacks(
            model_save_path=str(self.model_dir / 'best_fire_model.h5'),
            patience=early_stopping_patience
        )
        
        # Train model
        history = self.fire_cnn.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save training history
        self._save_training_history(history)
        
        print("Training completed!")
        
        return history
    
    def evaluate_model(self, test_data, save_results=True):
        """Evaluate model on test set"""
        
        if self.fire_cnn is None:
            raise ValueError("Model not trained yet.")
        
        X_test, y_test = test_data
        
        print("Evaluating model on test set...")
        
        # Create class names for evaluation
        class_names = {}
        for task, encoder in self.encoders.items():
            if task != 'size_category':
                # Reverse the encoder to get class names
                class_names[task] = [None] * len(encoder)
                for name, idx in encoder.items():
                    class_names[task][idx] = name
            else:
                # Size categories are predefined
                class_names[task] = ['Small', 'Medium', 'Large', 'Very Large']
        
        # Evaluate model
        results, predictions = self.fire_cnn.evaluate_model(X_test, y_test, class_names)
        
        if save_results:
            self._save_evaluation_results(results, predictions, y_test)
        
        return results, predictions
    
    def _save_training_history(self, history):
        """Save training history"""
        history_path = self.results_dir / 'training_history.json'
        
        # Convert history to serializable format
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        # Plot and save training curves
        self.fire_cnn.plot_training_history(
            save_path=str(self.results_dir / 'training_curves.png')
        )
    
    def _save_evaluation_results(self, results, predictions, y_test):
        """Save evaluation results"""
        
        # Save detailed results
        results_path = self.results_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save predictions
        predictions_path = self.results_dir / 'predictions.pkl'
        with open(predictions_path, 'wb') as f:
            pickle.dump({
                'predictions': predictions,
                'true_labels': y_test
            }, f)
        
        # Create and save confusion matrices
        self._create_confusion_matrices(predictions, y_test)
        
        print(f"Evaluation results saved to {self.results_dir}")
    
    def _create_confusion_matrices(self, predictions, y_test):
        """Create and save confusion matrices for each task"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        tasks = ['fire_type', 'ignition_cause', 'state', 'size_category']
        
        for i, task in enumerate(tasks):
            y_true = y_test[task]
            y_pred = np.argmax(predictions[task], axis=1)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                cbar_kws={'shrink': 0.8}
            )
            
            axes[i].set_title(f'{task.replace("_", " ").title()} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_training(self, gdb_path=None, sample_size=1000, 
                            architecture='custom', epochs=50, batch_size=32):
        """Run complete training pipeline from data processing to evaluation"""
        
        print("="*60)
        print("COMPLETE FIRE FINGERPRINTING TRAINING PIPELINE")
        print("="*60)
        
        # Step 1: Process data if needed
        if gdb_path and not self.data_dir.exists():
            print("Processing raw data...")
            processor = FireDataProcessor(gdb_path, output_dir=self.data_dir)
            fingerprints, labels, metadata = processor.process_fire_dataset(
                sample_size=sample_size
            )
        else:
            # Load existing processed data
            fingerprints, labels, metadata = self.load_processed_data()
        
        # Step 2: Prepare training data
        train_data, val_data, test_data = self.prepare_training_data(fingerprints, labels)
        
        # Step 3: Calculate class weights
        self.calculate_class_weights(train_data[1])
        
        # Step 4: Create model
        self.create_model(architecture=architecture)
        
        # Step 5: Train model
        history = self.train_model(
            train_data, val_data, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # Step 6: Evaluate model
        results, predictions = self.evaluate_model(test_data)
        
        # Step 7: Save final model and results
        self.fire_cnn.save_model(str(self.model_dir / 'final_fire_model.h5'))
        
        # Step 8: Generate summary report
        self._generate_summary_report(results, history)
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return self.fire_cnn, results
    
    def _generate_summary_report(self, results, history):
        """Generate comprehensive summary report"""
        
        report = {
            'training_date': datetime.now().isoformat(),
            'model_architecture': self.fire_cnn.model.name,
            'total_parameters': int(self.fire_cnn.model.count_params()),
            'training_epochs': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history['val_loss'][-1]),
            'task_accuracies': {}
        }
        
        # Add task-specific accuracies
        for task in ['fire_type', 'ignition_cause', 'state', 'size_category']:
            if task in results:
                report['task_accuracies'][task] = float(results[task]['accuracy'])
        
        # Calculate overall performance score
        accuracies = list(report['task_accuracies'].values())
        report['overall_accuracy'] = float(np.mean(accuracies))
        
        # Save report
        report_path = self.results_dir / 'training_summary.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\nTRAINING SUMMARY:")
        print(f"Model: {report['model_architecture']}")
        print(f"Parameters: {report['total_parameters']:,}")
        print(f"Training epochs: {report['training_epochs']}")
        print(f"Final training loss: {report['final_training_loss']:.4f}")
        print(f"Final validation loss: {report['final_validation_loss']:.4f}")
        print(f"Overall accuracy: {report['overall_accuracy']:.4f}")
        
        print(f"\nTask-specific accuracies:")
        for task, acc in report['task_accuracies'].items():
            print(f"  {task}: {acc:.4f}")
        
        print(f"\nResults saved to: {self.results_dir}")

def main():
    """Main training function"""
    
    # Initialize trainer
    trainer = FireTrainer()
    
    # Run complete training pipeline
    # Note: Adjust paths and parameters as needed
    gdb_path = "../data/Bushfire_Boundaries_Historical_2024_V3.gdb"
    
    model, results = trainer.run_complete_training(
        gdb_path=gdb_path,
        sample_size=1000,  # Start with sample for testing
        architecture='custom',
        epochs=30,  # Reduced for initial testing
        batch_size=16  # Smaller batch size for memory efficiency
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
