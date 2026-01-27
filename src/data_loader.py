import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class QuantumDataManager:
    def __init__(self, config):
        self.file_path = 'data/trajectories.csv'
        
        self.config = config
        
        self.input_width = config['window_size']
        self.forecast_horizon = config['forecast_horizon']
        self.batch_size = config['batch_size']
        
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def _split_trajectories(self, df):
        time_col = df.iloc[:, 0].values
        # Trova indici dove il tempo decresce (nuova traiettoria)
        split_indices = np.where(np.diff(time_col) < 0)[0] + 1
        traj_list = np.split(df.values, split_indices)
        
        # Rimuove la colonna tempo (col 0) e tiene le 55 feature fisiche
        clean_trajectories = [t[:, 1:] for t in traj_list]
        return clean_trajectories

    def _create_windowed_dataset(self, trajectories):
        """Crea le sequenze sliding window (X, y)."""
        X, y = [], []
        for t in trajectories:
            # Assicuriamoci che la traiettoria sia abbastanza lunga
            if len(t) > (self.input_width + self.forecast_horizon):
                for i in range(len(t) - self.input_width - self.forecast_horizon + 1):
                    X.append(t[i : i + self.input_width])
                    y.append(t[i + self.input_width : i + self.input_width + self.forecast_horizon])
        return np.array(X), np.array(y)

    def load_and_process(self, test_size=0.2, random_state=42):
        """
        Esegue la pipeline completa: Load -> Split Traj -> Train/Test Split -> Normalize -> Windowing
        
        Returns:
            X_train, y_train, X_test, y_test (numpy arrays pronti per il modello)
        """
        # 1. Caricamento
        df = pd.read_csv(self.file_path, header=None, index_col=False)
        print(f"Dataset caricato: {df.shape}")

        # 2. Split Traiettorie
        all_trajectories = self._split_trajectories(df)
        print(f"Traiettorie individuate: {len(all_trajectories)}")

        # 3. Train/Test Split (sulle traiettorie intere, non sui singoli punti!)
        train_traj, test_traj = train_test_split(
            all_trajectories, 
            test_size=test_size, 
            random_state=random_state
        )

        # 4. Fitting dello Scaler (SOLO su Train)
        # Concateniamo tutto il train per calcolare min e max globali
        train_concat = np.vstack(train_traj)
        self.scaler.fit(train_concat)

        # 5. Trasformazione
        train_traj_norm = [self.scaler.transform(t) for t in train_traj]
        test_traj_norm = [self.scaler.transform(t) for t in test_traj]

        # 6. Creazione Dataset Windowed
        self.X_train, self.y_train = self._create_windowed_dataset(train_traj_norm)
        self.X_test, self.y_test = self._create_windowed_dataset(test_traj_norm)

        print(f"[{self.config['name']}] Dataset caricato. Train shape: {self.X_train.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def get_tf_datasets(self):
        """
        Restituisce direttamente i dataset TensorFlow pronti per il training,
        utilizzando il batch_size definito nella config.
        """
        # Assicuriamoci che i dati siano caricati
        if not hasattr(self, 'X_train'):
            self.load_and_process()

        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        val_ds = val_ds.cache()
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds

    def inverse_transform(self, data):
        """
        Utile per riportare le predizioni ai valori fisici originali.
        Accetta dati di shape (batch, time, features) o (batch, features).
        """
        # Se i dati sono 3D (batch, time, feat), dobbiamo appiattirli per lo scaler e poi riformattarli
        original_shape = data.shape
        if len(original_shape) == 3:
            data_flat = data.reshape(-1, original_shape[2])
            data_inv = self.scaler.inverse_transform(data_flat)
            return data_inv.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)