import tensorflow as tf

class CurriculumLearning:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @staticmethod
    def shift_right(y):
        # Aggiunge un token di start (0) e toglie l'ultimo token
        start = tf.zeros_like(y[:, :1, :])
        return tf.concat([start, y[:, :-1, :]], axis=1)

    @tf.function(reduce_retracing=True)
    def teacher_forcing_step(self, x, y):
        y = tf.cast(y, tf.float32)
        
        # 1. Prepariamo l'input del decoder (Shiftato)
        decoder_input = CurriculumLearning.shift_right(y)
        
        with tf.GradientTape() as tape:
            preds = self.model(
                encoder_input=x,
                decoder_input=decoder_input,
                training=True
            )
            # 2. La Loss si calcola rispetto a Y originale (Target), non shiftato
            loss = self.loss_fn(y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    @tf.function(reduce_retracing=True)
    def masked_modeling_step(self, x, y, mask_prob=0.15):
        y = tf.cast(y, tf.float32)

        mask = tf.cast(
            tf.random.uniform(tf.shape(y)) < mask_prob,
            tf.float32
        )

        y_masked = y * (1.0 - mask)

        with tf.GradientTape() as tape:
            preds = self.model(
                encoder_input=x,
                decoder_input=y_masked,
                training=True
            )

            loss = tf.reduce_sum(
                self.loss_fn(y, preds) * mask
            ) / (tf.reduce_sum(mask) + 1e-9)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    @tf.function(reduce_retracing=True)
    def noisy_teacher_forcing_step(self, x, y, noise_std=0.1):
        """
        L'alternativa veloce allo Scheduled Sampling.
        Simula errori passati aggiungendo rumore all'input del decoder.
        """
        y = tf.cast(y, tf.float32)

        # 1. Creiamo l'input corretto (Teacher Forcing)
        decoder_input_clean = CurriculumLearning.shift_right(y)
        
        # 2. Aggiungiamo rumore SOLO all'input del decoder
        # Questo simula il fatto che il modello al passo precedente 
        # potrebbe aver predetto qualcosa di leggermente sbagliato.
        noise = tf.random.normal(shape=tf.shape(decoder_input_clean), stddev=noise_std)
        decoder_input_noisy = decoder_input_clean + noise

        with tf.GradientTape() as tape:
            preds = self.model(
                encoder_input=x,
                # Passiamo l'input "sporco"
                decoder_input=decoder_input_noisy, 
                training=True
            )
            # Ma calcoliamo la loss rispetto al target VERO e PULITO
            loss = self.loss_fn(y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
