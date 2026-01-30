import tensorflow as tf

class CurriculumLearning:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function(reduce_retracing=True)
    def teacher_forcing_step(self, x, y):
        y = tf.cast(y, tf.float32)
        
        with tf.GradientTape() as tape:
            preds = self.model(
                encoder_input=x,
                decoder_input=y,   # ground truth
                training=True
            )
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
    def noise_robustness_step(self, x, y, noise_std=0.01):
        y = tf.cast(y, tf.float32)

        noise = tf.random.normal(
            shape=tf.shape(x),
            mean=0.0,
            stddev=noise_std,
            dtype=tf.float32
        )
        x_noisy = x + noise

        with tf.GradientTape() as tape:
            preds = self.model(
                encoder_input=x_noisy,
                decoder_input=y,
                training=True
            )
            loss = self.loss_fn(y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
