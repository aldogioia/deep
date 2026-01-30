import tensorflow as tf
from keras import layers, Model

class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layerNorm = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context, training=False):
        attn_output = self.mha(query=x,value=context,key=context, training=training)

        x = self.add([x, attn_output])
        x = self.layerNorm(x)

        return x

class SelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x,value=x,key=x)

        x = self.add([x, attn_output])
        x = self.layerNorm(x)

        return x
    
class CausalSelfAttention(BaseAttention):
    def call(self, x, training=False):
        attn_output = self.mha(query=x,value=x,key=x, use_causal_mask=True, training=training)

        x = self.add([x, attn_output])
        x = self.layerNorm(x)

        return x
    
class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(d_ff, activation="relu"),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        self.layerNorm = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()

    def call(self, x):
        ffn_output = self.seq(x)

        x = self.add([x, ffn_output])
        x = self.layerNorm(x)

        return x

class LearnablePositionalEncoding(layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, seq_len, d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        curr_seq_len = tf.shape(x)[1]
        
        return x + self.pos_emb[:, :curr_seq_len, :]
    
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = SelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def call(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
    def call(self, x, enc_out, training=False):
        x = self.causal_self_attention(x, training=training)
        x = self.cross_attention(x, enc_out, training=training)
        x = self.feed_forward(x)
        return x

class QuantumTransformer(Model):
    def __init__(self, input_dim, seq_len, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        self.enc_input_proj = layers.Dense(d_model)
        self.dec_input_proj = layers.Dense(d_model)

        self.enc_pos = LearnablePositionalEncoding(seq_len, d_model)
        self.dec_pos = LearnablePositionalEncoding(seq_len, d_model)

        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

        self.output_head = layers.Dense(input_dim)

    def call(self, encoder_input, decoder_input, training=False):
        encoder_output = self.enc_input_proj(encoder_input)
        encoder_output = self.enc_pos(encoder_output)

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        decoder_output = self.dec_input_proj(decoder_input)
        decoder_output = self.dec_pos(decoder_output)

        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output,
                encoder_output,
                training=training
            )
        
        return self.output_head(decoder_output)