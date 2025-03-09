import keras
from keras import ops
from keras import layers
from keras.models import load_model
import numpy as np

# implement a transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self,embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    
    def call(self, inputs):
        attn_output = self.att(query=inputs, key=inputs, value=inputs)  
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output) 

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
        })
        return config

#implemnt embedding layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self,maxlen,vocab_size,embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)


    def call(self,x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.pos_emb.input_dim,
            "vocab_size": self.token_emb.input_dim,
            "embed_dim": self.token_emb.output_dim,
        })
        return config


#download and prepare dataset
vocab_size = 3000
maxlen = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)


print(x_train.shape)
print(y_train.shape)



print(len(x_train), "Training Sequences") # is a numpy array 
#print(x_train[0])
print(len(x_val), "Validation Sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

#print("x_train length" + str(len(x_train)))

#create classifier model using transformer layer 
embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers. Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print("---")
model.summary()
print("---")

#train and evlaluate

train_model = True
if train_model == True:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val)
    )





def save_model(model, filename="transformer_model.keras"):
    model.save(filename)
    print(f"Model saved to {filename}")




def load_trained_model(filename="transformer_model.keras"):
    model = load_model(filename, custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    })
    print(f"Model loaded from {filename}")
    return model


save_model(model)
#model = load_trained_model()


def predict_sentiment(model, tokenized_input):
    #print(f"Model Input Shape: {tokenized_input.shape}")  # Should be (1, 200)

    # Make prediction
    prediction = model.predict(tokenized_input)
    predicted_label = np.argmax(prediction)  # 0 = negative, 1 = positive
    confidence = np.max(prediction)

    sentiment = "Positive" if predicted_label == 1 else "Negative"
    print(text_input + f" - Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    return predicted_label, confidence


word_index = keras.datasets.imdb.get_word_index()

index_to_word = {index + 3: word for word, index in word_index.items()}  # Keras reserves index 0,1,2
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"


def tokenize_text(text, word_index, maxlen=200):
    words = text.lower().split()  # Convert text to lowercase and split into words
    sequence = []
    
    #print("\nTokenizing Sentence: ", text)  # Debugging Output
    
    for word in words:
        token = word_index.get(word, 2) + 3  # Apply the +3 offset for unknown words
        sequence.append(token)
        #print(f"Word: {word} → Token: {token}")  # Print word-token mapping
    
    # Pad sequence to ensure it is the same length as training data
    padded_sequence = keras.utils.pad_sequences([sequence], maxlen=maxlen, padding="pre")

    #print(f"Final Tokenized Sequence: {padded_sequence}\n")
    return padded_sequence


text_input = "This was the best movie I have ever seen. I loved every moment of it!"
"""
    "I really enjoyed this film. The characters were well-developed and the story was captivating.",
    "What a waste of time! The plot made no sense and the acting was terrible.",
    "I hated this movie. It was boring and the dialogue was so bad.",
    "The cinematography was beautiful, but the story lacked depth.",
    "This was the best movie I have ever seen. I loved every moment of it!",
    "Awful experience. The movie was too long, and I almost fell asleep.",
    "Surprisingly good! I went in with low expectations but ended up really enjoying it.",
    "The script was poorly written, and the jokes were not funny at all.",
    "An emotional rollercoaster. I laughed, I cried, and I was completely invested in the characters."
"""


tokenized_input = tokenize_text(text_input, word_index, maxlen=maxlen)


# Predict sentiment
predict_sentiment(model, tokenized_input)