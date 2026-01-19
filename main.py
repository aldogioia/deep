from tensorflow.keras import layers, Model

class QuantumRNN(Model):
    """
    La classe QuantumRNN implementa un'architettura neurale ricorrente
    basata su GRU, progettata specificamente per la previsione della
    dinamica di sistemi quantistici (Qubits) soggetti a decoerenza.

    Architettura:
    - LayerNormalization:
        Normalizza le feature di input per ogni timestep. Poiché i dati
        del dataset Quantum (magnetizzazioni e correlazioni) possono avere
        scale e fluttuazioni diverse a seconda del tempo di simulazione,
        la normalizzazione garantisce che i gradienti rimangano stabili
        e che nessuna feature domini le altre durante l'apprendimento.

    - GRU a due livelli:
        - Il primo layer GRU elabora la sequenza temporale di input e
        restituisce l'intera sequenza di stati nascosti. Questo permette
        al secondo layer di analizzare una rappresentazione gerarchica più
        ricca della dinamica temporale, estraendo pattern complessi dalle
        interazioni tra qubit.

        - Il secondo layer comprime le informazioni temporali, restituendo
        solo l'ultimo stato nascosto. Rappresenta il riassunto dell'intera
        evoluzione passata del sistema, pronto per essere mappato nello
        stato futuro.

    - Dense finale:
        Layer fully-connected che mappa lo stato latente finale in uno spazio
        di output continuo di dimensione `output_dim`. Essendo un problema di
        regressione, non viene applicata alcuna funzione di attivazione
        (come Sigmoid o Softmax), permettendo al modello di predire valori
        continui nell'intervallo reale richiesto per descrivere magnetizzazioni
        e correlazioni.
    """

    def __init__(self, hidden_units, output_dim, dropout_rate=0.2):
        super(QuantumRNN, self).__init__()

        # Layer di Normalizzazione
        self.norm = layers.LayerNormalization()

        # GRU 1: return_sequences=True per passare la sequenza alla seconda GRU
        # Aggiungiamo dropout e recurrent_dropout per regolarizzazione
        self.gru1 = layers.GRU(
            hidden_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        )

        # GRU 2: Elabora la sequenza e restituisce solo l'ultimo stato
        self.gru2 = layers.GRU(
            hidden_units,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        )

        # Dense Layer finale per la regressione (56 parametri)
        self.dense_out = layers.Dense(output_dim)

    def call(self, inputs, training=False):
        # Il parametro 'training' gestisce automaticamente il dropout:
        # viene attivato solo durante fit/train e disattivato durante l'esame (inferenza)
        x = self.norm(inputs)
        x = self.gru1(x, training=training)
        x = self.gru2(x, training=training)
        return self.dense_out(x)