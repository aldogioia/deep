import os
import pandas as pd
import matplotlib.pyplot as plt

class ExperimentManager:
    def __init__(self, base_path="data"):
        self.weights_path = os.path.join(base_path, "weights")
        self.history_path = os.path.join(base_path, "history")
        self.plots_path = os.path.join(base_path, "plots")
        
        # Crea le cartelle se non esistono
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def save_model_artifacts(self, model, history, config_name):
        """Salva i pesi in .h5 e la history in .csv"""
        
        # 1. Salvataggio Pesi
        weight_file = os.path.join(self.weights_path, f"{config_name}.weights.h5")
        model.save_weights(weight_file)
        print(f"Pesi salvati in: {weight_file}")
        
        # 2. Salvataggio History
        hist_df = pd.DataFrame(history)
        hist_file = os.path.join(self.history_path, f"{config_name}_history.csv")
        hist_df.to_csv(hist_file, index=False)
        print(f"History salvata in: {hist_file}")

    def plot_loss_curves(self, history, config_name):
        """Plotta Train vs Validation Loss e le fasi del curriculum"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(history['loss']) + 1)
        plt.plot(epochs, history['loss'], label='Training Loss', linewidth=2)
        
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
            
        # Disegna linee verticali per i cambi di fase
        # Cerchiamo dove cambia la colonna 'phase' nel dataframe o dict
        phases = history['phase']
        for i in range(1, len(phases)):
            if phases[i] != phases[i-1]:
                plt.axvline(x=i+1, color='gray', linestyle=':', alpha=0.7)
                plt.text(i+1, max(history['loss']), phases[i], rotation=90, verticalalignment='top')

        plt.title(f'Loss Curves - {config_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.join(self.plots_path, f"{config_name}_loss.png")
        plt.savefig(plot_file)
        plt.show()

    def plot_forecast_comparison(self, model, val_ds, data_manager, config_name, num_samples=3):
        """
        Prende alcuni campioni dal validation set, fa la predizione e plotta 
        Input (History) + Target (Ground Truth) + Prediction
        """
        # Estrai un batch dal dataset
        x_val, y_val = next(iter(val_ds))
        
        # Predizione (Teacher Forcing off -> Inference pura sarebbe meglio, 
        # ma qui usiamo simple forward per coerenza col training o loop autoregressivo)
        # Nota: Per visualizzare bene, facciamo una predizione "one-step-ahead" style 
        # o idealmente autoregressiva. Qui usiamo il forward pass standard del transformer.
        preds = model(x_val, y_val, training=False)
        
        # Converti in numpy
        x_val = x_val.numpy()
        y_val = y_val.numpy()
        preds = preds.numpy()
        
        # Denormalizza
        x_real = data_manager.inverse_transform(x_val)
        y_real = data_manager.inverse_transform(y_val)
        p_real = data_manager.inverse_transform(preds)
        
        input_width = x_real.shape[1]
        forecast_horizon = y_real.shape[1]
        
        # Plot di N campioni
        for i in range(num_samples):
            plt.figure(figsize=(12, 4))
            
            # Feature 0 (es. Posizione X)
            feat_idx = 0 
            
            # Input History
            time_in = range(input_width)
            plt.plot(time_in, x_real[i, :, feat_idx], 'b-', label='History')
            
            # Ground Truth
            time_out = range(input_width, input_width + forecast_horizon)
            plt.plot(time_out, y_real[i, :, feat_idx], 'g-', label='Ground Truth')
            
            # Prediction
            plt.plot(time_out, p_real[i, :, feat_idx], 'r--', label='Prediction')
            
            plt.title(f"Sample {i+1} - {config_name}")
            plt.legend()
            
            plot_file = os.path.join(self.plots_path, f"{config_name}_pred_{i}.png")
            plt.savefig(plot_file)
            plt.show()