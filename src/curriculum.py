import tensorflow as tf

class CurriculumLearning:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function(reduce_retracing=True)
    def teacher_forcing_step(self, x, y):
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
    def scheduled_sampling_step(self, x, y, sampling_prob):
        batch, T, d = tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2]

        dec_in = tf.zeros_like(y[:, :1, :])  # start token (zero)

        preds_all = []

        with tf.GradientTape() as tape:
            for t in range(T):
                preds = self.model(
                    encoder_input=x,
                    decoder_input=dec_in,
                    training=True
                )

                next_pred = preds[:, -1:, :]

                use_model = tf.random.uniform((batch, 1, 1)) < sampling_prob
                next_input = tf.where(
                    use_model,
                    next_pred,
                    y[:, t:t+1, :]
                )

                dec_in = tf.concat([dec_in, next_input], axis=1)
                preds_all.append(next_pred)

            preds_all = tf.concat(preds_all, axis=1)

            loss = self.loss_fn(y, preds_all)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    
