import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def __init__(self, base_path="data"):
        self.weights_path = os.path.join(base_path, "weights")
        self.history_path = os.path.join(base_path, "history")
        self.plots_path = os.path.join(base_path, "plots")
        os.makedirs(self.weights_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)

    def load_model_weights(self, model, config_name):
        """Carica i pesi salvati senza dover riaddestrare"""
        weight_file = os.path.join(self.weights_path, f"{config_name}.weights.h5")
        
        if os.path.exists(weight_file):
            print(f"Trovato file pesi: {weight_file}")
            
            # --- FIX: Recuperiamo le dimensioni dagli attributi del modello ---
            # 1. Recuperiamo la dimensione delle feature dal layer di output (units)
            #    che è stato inizializzato come layers.Dense(input_dim)
            feature_dim = model.output_head.units 
            
            # 2. Recuperiamo la lunghezza sequenza salvata nel modello
            seq_len = model.seq_len
            
            # 3. Creiamo input dummy corretti
            # Nota: Usiamo float32 per evitare conflitti di tipo
            dummy_x = tf.zeros((1, seq_len, feature_dim), dtype=tf.float32)
            dummy_y = tf.zeros((1, seq_len, feature_dim), dtype=tf.float32)
            
            # 4. Chiamata a vuoto per inizializzare i pesi (Build)
            print("Inizializzazione pesi (Dummy Call)...")
            model(dummy_x, dummy_y, training=False)
            
            # 5. Caricamento effettivo
            model.load_weights(weight_file)
            print("Pesi caricati correttamente.")
        else:
            print(f"ERRORE: File pesi non trovato in {weight_file}")
            print("Verifica di aver eseguito il training e salvato i pesi con lo stesso 'name' nella config.")

    def plot_training_phases_detailed(self, history, config_name):
        """
        Plotta Loss e Val Loss rilevando automaticamente i cambi di fase dalla history.
        """
        loss = history['loss']
        val_loss = history.get('val_loss', [])
        phases = history['phase']
        epochs = range(1, len(loss) + 1)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # --- PLOT CURVE ---
        ax1.plot(epochs, loss, label='Train Loss', color='#1f77b4', linewidth=2)
        if val_loss:
            ax1.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=2)

        # --- RILEVAMENTO FASI DINAMICO ---
        # Trova gli indici in cui la fase cambia
        phase_changes = [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
        
        # Aggiungi inizio (0) e fine (len) per calcolare i centri
        boundaries = [0] + phase_changes + [len(phases)]
        
        y_min, y_max = ax1.get_ylim()
        text_y_pos = y_max - (y_max - y_min) * 0.05

        # Disegna linee e testo per ogni fase
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            phase_name = phases[start].replace('_', '\n').upper() # Es: TEACHER\nFORCING
            
            # Centro della fase per il testo
            mid_point = (start + end) / 2 + 0.5 
            
            # Testo identificativo
            ax1.text(mid_point, text_y_pos, phase_name, ha='center', va='top', 
                     fontsize=9, fontweight='bold', color='gray', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
            
            # Linea verticale separatrice (solo se non è l'ultimo punto)
            if end != len(phases):
                ax1.axvline(x=end + 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        ax1.set_title(f'Training Dynamics - {config_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoche')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_path, f"training/loss_{config_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"Grafico salvato in: {plot_path}")

    def plot_trajectory_check(self, model, data_manager, config_name, num_steps=1000):
        """
        Plotta una traiettoria lunga e continua confrontando Realtà vs Predizione.
        
        Args:
            num_steps: Quanti step temporali visualizzare (es. 1000 per replicare la tua immagine).
        """
        # 1. Recuperiamo i dati di Test
        # X_test ha shape (N_samples, window_size, features)
        # y_test ha shape (N_samples, horizon, features)
        # Assumiamo che i dati nel test set siano sequenziali (stride=1)
        X_data = data_manager.X_test
        y_data = data_manager.y_test
        
        # Limitiamo ai primi 'num_steps' per non appesantire il plot
        limit = min(len(X_data), num_steps)
        
        # Selezioniamo il chunk di dati
        x_chunk = X_data[:limit] # (limit, window, feat)
        y_chunk = y_data[:limit] # (limit, horizon, feat)
        
        print(f"Generazione predizioni per {limit} step...")
        
        # 2. Predizione in Batch (Molto più veloce del ciclo for)
        # Il Transformer richiede (encoder_input, decoder_input)
        # Usiamo Teacher Forcing (passiamo y_chunk come decoder input) per vedere
        # quanto bene il modello fitta la curva passo-passo.
        
        # Convertiamo in tensori
        x_tensor = tf.convert_to_tensor(x_chunk, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_chunk, dtype=tf.float32)
        
        # Inferenza (training=False spegne il Dropout)
        preds_tensor = model(encoder_input=x_tensor, decoder_input=y_tensor, training=False)
        preds_numpy = preds_tensor.numpy() # (limit, horizon, features)
        
        # 3. Estrazione della linea continua
        # X_test è creato con sliding window di passo 1.
        # Quindi X[0] predice t+1, X[1] predice t+2, ecc.
        # Prendiamo solo il PRIMO step di ogni predizione (One-Step-Ahead) per formare la linea.
        
        # Ground Truth: Il primo step del target window
        truth_seq = y_chunk[:, 0, :] # Shape (limit, features)
        
        # Prediction: Il primo step della finestra predetta
        pred_seq = preds_numpy[:, 0, :] # Shape (limit, features)
        
        # 4. Denormalizzazione (Inverse Transform)
        truth_real = data_manager.inverse_transform(truth_seq)
        pred_real = data_manager.inverse_transform(pred_seq)
        
        # 5. Plotting (Stile della tua immagine di riferimento)
        plt.figure(figsize=(16, 6)) # Largo come richiesto
        
        feature_idx = 0 # Feature principale (es. Magnetizzazione o Posizione)
        
        # Ground Truth: Linea Nera Solida (trasparenza leggera per vedere sotto)
        plt.plot(truth_real[:, feature_idx], 
                 label='Realtà (Ground Truth)', 
                 color='black', 
                 alpha=0.8, 
                 linewidth=1.8)
        
        # Prediction: Linea Rossa Tratteggiata
        plt.plot(pred_real[:, feature_idx], 
                 label=f'Predizione ({config_name})', 
                 color='#d62728', # Rosso standard matplotlib
                 linestyle='--', 
                 linewidth=1.5)
        
        plt.title(f'Verifica Traiettoria: {config_name} (Samples: {limit})', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Valore Fisico')
        plt.legend(loc='upper right')
        
        # Griglia leggera come nell'immagine
        plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
        
        plot_path = os.path.join(self.plots_path, f"predictions/pred_{config_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Grafico salvato in: {plot_path}")

    def plot_attention_map(self, model, data_manager, config_name, sample_idx=0):
        import seaborn as sns
        
        # Preparazione dati
        encoder_input = data_manager.X_test[sample_idx : sample_idx+1]
        decoder_input = data_manager.y_test[sample_idx : sample_idx+1]
        
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.float32)
        decoder_input = tf.convert_to_tensor(decoder_input, dtype=tf.float32)

        print("Recupero mappe di attenzione...")
        try:
            _, attn_weights = model(encoder_input, decoder_input, training=False, return_attention=True)
        except Exception as e:
            print(f"Errore nel recupero pesi: {e}")
            return

        # Creiamo una figura con 2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # --- 1. ENCODER SELF-ATTENTION ---
        if 'encoder_self_attn' in attn_weights:
            enc_weights = tf.reduce_mean(attn_weights['encoder_self_attn'], axis=1)[0].numpy()
            
            sns.heatmap(enc_weights, cmap='viridis', ax=axes[0], square=True, cbar=True)
            axes[0].set_title(f"Encoder Self-Attention (Input vs Input)\nConfig: {config_name}")
            axes[0].set_xlabel("Input Key (Time t)")
            axes[0].set_ylabel("Input Query (Time t)")
        else:
            axes[0].text(0.5, 0.5, "Encoder Attention Not Found", ha='center')

        # --- 2. DECODER CROSS-ATTENTION ---
        if 'decoder_cross_attn' in attn_weights:
            dec_weights = tf.reduce_mean(attn_weights['decoder_cross_attn'], axis=1)[0].numpy()
            
            sns.heatmap(dec_weights, cmap='viridis', ax=axes[1], square=False, annot=True, cbar=True)
            axes[1].set_title(f"Decoder Cross-Attention (Prediction vs History)\nConfig: {config_name}")
            axes[1].set_xlabel(f"History Steps (t-{len(dec_weights[0])} ... t)")
            axes[1].set_ylabel("Prediction Step (t+1 ...)")
        else:
            axes[1].text(0.5, 0.5, "Decoder Attention Not Found", ha='center')

        plt.tight_layout()
        plot_path = os.path.join(self.plots_path, f"attentions/att_{config_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"Grafico combinato salvato in: {plot_path}")