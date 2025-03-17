import keras
from keras import ops
from keras import layers
from keras.models import load_model
import numpy as np
import tensorflow as tf
import random
import wandb
from wandb.integration.keras import WandbMetricsLogger
from generate_experimental_data_set import load_data



wandb.login()

learning_rate = 0.001
epochs = 10
#setting up wandb

wandb.init(config={"bs":12})






# implement a transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        """Initialize transformer block.
        
        Args:
            embed_dim: Size of embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden layer size in feed forward network
            rate: Dropout rate
            **kwargs: Additional layer arguments
        """
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
        """Apply transformer block to input.
        
        Args:
            inputs: Input tensor of shape `(batch_size, seq_len, embed_dim)`
            
        Returns:
            Tensor of same shape after self-attention and feed-forward processing
        """
        attn_output = self.att(query=inputs, key=inputs, value=inputs)  
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output) 

    def get_config(self):
        """Returns layer configuration for serialization.
        
        Returns:
            dict: Configuration with embedding dimension, number of attention heads,
                 and feed-forward layer dimension.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
        })
        return config

#implemnt embedding layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        """Initialize token and position embedding layer.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Size of the vocabulary
            embed_dim: Size of the embedding dimension
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)


    def call(self,x):
        """Add positional embeddings to token embeddings.
        
        Args:
            x: Integer tensor of token indices of shape `(batch_size, seq_len)`
            
        Returns:
            Tensor of shape `(batch_size, seq_len, embed_dim)` containing token+position embeddings
        """
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        """Returns layer configuration for serialization.
        
        Returns:
            dict: Configuration with maxlen, vocab_size, and embedding dimension.
        """
        config = super().get_config()
        config.update({
            "maxlen": self.pos_emb.input_dim,
            "vocab_size": self.token_emb.input_dim,
            "embed_dim": self.token_emb.output_dim,
        })
        return config


#download and prepare dataset
vocab_size = 130000
maxlen = 200

(x_train, y_train), (x_val, y_val) = load_data(num_words=vocab_size)

#(x_train, y_train), (x_val, y_val) = generate_sample.load_data(num_words=vocab_size)

print(len(x_train), "Training Sequences") # is a numpy array 
#print(x_train[0])
print(len(x_val), "Validation Sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

#print("x_train length" + str(len(x_train)))

#create classifier model using transformer layer 
# Enhanced model parameters
embed_dim = 128  # Larger embedding for better context
num_heads = 8    # More attention heads for nuanced understanding
ff_dim = 256     # Larger feed-forward for complex patterns

# Model architecture
inputs = layers.Input(shape=(maxlen,))

# Enhanced embedding with larger dimension
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)

# Multiple transformer blocks with residual connections
transformer_blocks = [
    TransformerBlock(embed_dim, num_heads, ff_dim),
    TransformerBlock(embed_dim, num_heads, ff_dim),
    TransformerBlock(embed_dim, num_heads, ff_dim)
]

# Apply transformer blocks with residual connections
for block in transformer_blocks:
    block_output = block(x)
    x = layers.Add()([x, block_output])  # Residual connection
    x = layers.LayerNormalization(epsilon=1e-6)(x)

# Temporal attention for time-dependent features
temporal_attention = layers.MultiHeadAttention(
    num_heads=4, key_dim=embed_dim//4)(x, x)
x = layers.Add()([x, temporal_attention])

# Global context with attention pooling
attention = layers.Dense(1, activation='tanh')(x)
attention_weights = layers.Softmax(axis=1)(attention)
x = layers.Multiply()([x, attention_weights])
x = layers.GlobalAveragePooling1D()(x)

# Advanced classification head
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

# Parallel sentiment and neutrality detection
sentiment_branch = layers.Dense(64, activation='relu')(x)
sentiment_branch = layers.Dropout(0.2)(sentiment_branch)
sentiment_output = layers.Dense(2, activation='softmax', name='sentiment')(sentiment_branch)

neutrality_branch = layers.Dense(64, activation='relu')(x)
neutrality_branch = layers.Dropout(0.2)(neutrality_branch)
neutrality_output = layers.Dense(1, activation='sigmoid', name='neutrality')(neutrality_branch)

# Combine outputs
outputs = [sentiment_output, neutrality_output]

model = keras.Model(inputs=inputs, outputs=outputs)


def save_model(model, filename="testing_experimntal_data"):
    model.save(filename)
    print(f"Model saved to {filename}")




def load_trained_model(filename="testing_experimntal_data"):
    model = load_model(filename, custom_objects={
        "TransformerBlock": TransformerBlock,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding
    })
    print(f"Model loaded from {filename}")
    return model

#train and evlaluate

train_model = False  # Set to True to train the model
if train_model:
    # Compile model with multiple outputs
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'sentiment': 'categorical_crossentropy',
            'neutrality': 'binary_crossentropy'
        },
        loss_weights={
            'sentiment': 1.0,
            'neutrality': 0.5
        },
        metrics={
            'sentiment': ['accuracy'],
            'neutrality': ['accuracy']
        }
    )
    
    # Convert labels for training
    y_sentiment = keras.utils.to_categorical(y_train, num_classes=2)
    y_neutrality = np.zeros_like(y_train, dtype=float)  # Initialize neutrality labels
    
    # Mark samples as neutral based on confidence threshold
    neutral_mask = np.random.random(len(y_train)) < 0.2  # 20% of samples marked as neutral
    y_neutrality[neutral_mask] = 1
    
    # Train the model
    history = model.fit(
        x_train,
        {'sentiment': y_sentiment, 'neutrality': y_neutrality},
        batch_size=32,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1
            ),
            WandbMetricsLogger()
        ]
    )
    




    # Save the trained model
    save_model(model, "transformer_model_enhanced.keras")
else:
    # Load existing model
    model = load_trained_model("transformer_model_enhanced.keras")










def predict_sentiment(model, tokenized_input):
    # Make prediction
    sentiment_pred, neutrality_pred = model.predict(tokenized_input)
    
    # Get sentiment probabilities
    neg_prob, pos_prob = sentiment_pred[0]
    neutrality_score = float(neutrality_pred[0][0])
    
    # Calculate sentiment strength and ambiguity
    sentiment_strength = abs(pos_prob - neg_prob)
    ambiguity_score = 1 - sentiment_strength
    
    # Determine final classification using both outputs
    if neutrality_score > 0.5:  # Strong neutrality signal
        sentiment = "Neutral"
        confidence = neutrality_score
    else:  # Use sentiment prediction
        sentiment = "IS an action potential" if pos_prob > neg_prob else "IS NOT an action potential"
        confidence = max(pos_prob, neg_prob)
    
    # Add detailed analysis
    print(f"Analysis for: {text_input}")
    print(f"Predicted: {sentiment}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Ambiguity Score: {ambiguity_score:.2%}")
    print(f"Detailed Probabilities:")
    print(f"  not an action potential: {neg_prob:.2%}")
    print(f"  is an action potential: {pos_prob:.2%}")
    print(f"  Neutrality: {neutrality_score:.2%}")
    
    # Additional analysis for specific cases
    if neutrality_score > 0.3 and sentiment_strength < 0.4:
        print("Note: analysis contains balanced or neutral content")
    if ambiguity_score > 0.4:
        print("Note: High ambiguity detected in analysis")
    if abs(neutrality_score - 0.5) < 0.1 and sentiment_strength < 0.3:
        print("Note: analysis may contain mixed or time-dependent opinions")
    
    return sentiment, confidence, ambiguity_score


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

text_input = """This is the neurons morphology: 
Soma length: 12.6157 meters
Soma diameter: 12.6157 meters
Dendrite length: 200 meters
Dendrite diameter: 1 meter
This is the neurons biophysics: 
Axial resistance: 100 Ohm*cm 
Membrane capacitance: 1 microfarads/cm^2
Sodium Conductance: 0.12 siemens
Potassium Conductance: 0.036 siemens 
Leak conductance: 0.003 siemens 
Reversal potential: -54.3 millivolts 
Passive conductance: 0.001 siemens
Leak reversal potential: -65 millivolts 
Ultrasound stimulation: 
Pressure points:[1998.3013, 1996.1659, 1964.1837, 1900.7778, 1793.1914, 1631.5813, 1423.1565, 1165.3794, 901.4516, 674.95465, 475.24542, 306.74988, 176.80333, 61.077156, -59.359386, -167.67894, -293.1625, -437.7399, -580.99, -719.4849, -824.75757, -841.853, -790.8091, -691.35266, -557.9762, -391.11536, -232.37038, -81.50432, 52.86182, 209.37064, 397.87317, 599.59283, 814.4537, 1021.54144, 1215.6447, 1390.2274, 1541.1224, 1695.6377, 1838.943, 1993.2839, 2149.437, 2278.3499, 2406.1902, 2544.955, 2691.1824, 2858.1084, 3029.602, 3180.9236, 3286.6733, 3333.9417, 3326.3123, 3296.9514, 3267.2678, 3243.7026, 3214.803, 3117.6904, 2938.5361, 2682.437, 2387.7373, 2103.4297, 1811.7761, 1522.3981, 1201.4137, 836.3025, 447.35245, 77.47956, -233.89394, -498.3056, -759.1448, -1008.4047, -1250.2284, -1413.2714, -1502.7435, -1523.2684, -1498.5876, -1448.1318, -1359.3888, -1215.8064, -992.4795, -671.47046, -295.44635, 43.59722, 333.4606, 576.23834, 814.15356, 1085.0333, 1397.1113, 1731.1869, 2041.138, 2252.0964, 2356.337, 2373.1228, 2388.6135, 2470.7532, 2618.2495, 2725.1035, 2757.0652, 2722.431, 2641.3276, 2578.9824, 2544.7854, 2500.9207, 2416.4988, 2279.2283, 2074.7744, 1928.6494, 1855.2618, 1855.9406, 1877.541, 1848.5686, 1781.8964, 1671.2793, 1584.6467, 1462.5876, 1370.4465, 1294.1511, 1191.3787, 1082.7551, 979.53925, 877.3655, 825.5116, 767.8787, 717.83185, 566.576, 359.8138, 136.65749, -40.27637, -115.948654, -114.15407, -95.2527, -67.75511, -71.44997, -58.613064, 3.1898816, 147.3189, 287.43292, 398.3194, 423.71835, 372.515, 351.64014, 400.48285, 573.6984, 809.761, 1010.0434, 1103.0815, 1117.5896, 1142.6381, 1244.6969, 1431.333, 1674.015, 1880.4833, 1959.3777, 1926.2612, 1869.5796, 1840.4095, 1897.1433, 2041.9272, 2207.9058, 2291.9216, 2278.6638, 2192.5544, 2107.7751, 2052.3892, 2067.4463, 2075.1648, 2012.2544, 1887.9739, 1722.8896, 1566.8402, 1478.4957, 1420.6146, 1375.1824, 1245.8907, 1034.4999, 782.8878, 566.90875, 422.9402, 335.0128, 273.51392, 154.05765, -15.84458, -207.70052, -321.72083, -353.75305, -333.07492, -305.09448, -311.18814, -347.8812, -373.40564, -329.53314, -203.59296, -80.89916, -37.181187, -107.54957, -235.14096, -297.51846, -251.80524, -89.01077, 73.33, 163.4588, 121.996216, 51.359474, 30.58345, 104.55854, 325.48013, 601.9394, 784.83386, 854.30725, 845.419, 873.29205, 1050.957, 1351.7975, 1694.3763, 1999.0337, 2187.5986, 2260.345, 2321.8425, 2440.3433, 2629.1814, 2864.5386, 3096.6162, 3236.4194, 3256.6628, 3244.9924, 3232.5066, 3223.0603, 3217.9583, 3162.836, 3054.7393, 2918.5671, 2744.9224, 2521.9858, 2236.2178, 1919.7334, 1622.742, 1330.0593, 1052.5557, 804.2005, 547.26196, 228.70433, -119.30201, -453.24158, -711.3612, -858.5553, -919.63995, -974.0718, -1093.2979, -1273.3391, -1482.0273, -1635.5287, -1676.9102, -1577.965, -1381.5162, -1177.3805, -1043.1816, -1048.5641, -1158.3821, -1246.1642, -1244.4938, -1123.2556, -936.95605, -758.5594, -588.2912, -454.8453, -331.30002, -214.31607, -90.52278, 24.150644, 97.43761, 146.55263, 208.36176, 302.68777, 425.6775, 555.51965, 626.6155, 629.3615, 610.649, 611.94946, 684.57086, 818.35864, 969.45355, 1096.2954, 1170.9979, 1202.459, 1215.3662, 1262.1366, 1363.9849, 1547.4814, 1763.3163, 1943.2335, 2002.0009, 1981.9427, 1967.2235, 2043.0979, 2202.2031, 2396.3093, 2583.299, 2721.3728, 2777.8696, 2757.9036, 2676.617, 2616.8813, 2659.967, 2830.5068, 3032.9429, 3183.597, 3248.603, 3227.8572, 3176.5725, 3139.8909, 3090.0388, 3039.382, 2999.9912, 2950.2014, 2898.9167, 2854.8423, 2810.2554, 2748.6667, 2631.829, 2439.7886, 2227.286, 2063.5637, 2020.5526, 2092.7466, 2168.6133, 2155.2954, 2054.9663, 1900.2988, 1775.408, 1802.5736, 2038.8318, 2438.7756, 2921.5159, 3397.1323, 3862.1604, 4346.587, 4953.8228, 5616.333, 6196.281, 6506.237, 6438.426, 6005.609, 5390.7344, 4779.006, 4258.5845, 3844.557, 3416.7148, 2787.1794, 1847.902, 611.22656, -859.49695, -2483.9236, -4171.64, -5892.511, -7592.8223, -9191.0, -10621.494, -11873.241, -12963.138, -13949.337, -14829.582, -15611.46, -16173.896, -16414.328, -16205.669, -15576.064, -14662.597, -13697.315, -12823.23, -12052.315, -11340.277, -10561.811, -9681.706, -8681.909, -7613.085, -6461.572, -5282.28, -4102.585, -2973.9058, -1972.4187, -1181.922, -670.61584, -427.2404, -350.06653, -328.87018, -292.7733, -235.76498, -181.72041, -188.90445, -247.03294, -352.91983, -415.24844, -367.6311, -212.53102, -21.95418, 112.4655, 115.45996, 56.434536, 20.983988, 80.54732, 241.66681, 433.44818, 590.55505, 678.4938, 720.988, 779.8311, 872.33136, 987.60693, 1073.9215, 1090.1185, 1059.118, 1055.5531, 1144.0458, 1362.7944, 1650.6353, 1908.3158, 2045.9733, 2032.0404, 1935.6759, 1859.2812, 1870.2549, 1950.8586, 2018.7106, 1963.662, 1782.0715, 1556.3286, 1378.3104, 1291.3143, 1266.3328, 1189.5664, 1007.2854, 687.96106, 263.07974, -152.10161, -430.58197, -561.17706, -619.2142, -751.1953, -1044.8049, -1444.6307, -1778.4214, -1923.1967, -1848.048, -1667.3516, -1476.7996, -1352.9775, -1294.9736, -1270.9744, -1207.0679, -1046.1947, -805.53076, -570.7666, -435.6298, -468.0992, -643.27136, -875.94727, -1055.6167, -1130.332, -1116.0579, -1097.0043, -1205.1891, -1496.5605, -1918.5686, -2369.824, -2749.089, -3039.861, -3288.0664, -3542.2625, -3826.3894, -4141.615, -4455.587, -4691.6855, -4798.0703, -4725.6235, -4558.579, -4379.614, -4280.9775, -4248.26, -4226.834, -4156.686, -4021.7512, -3851.205, -3698.349, -3582.8728, -3504.1123, -3425.871, -3314.7498, -3179.2117, -3073.6135, -3055.9238, -3116.511, -3212.483, -3296.7395, -3329.7302, -3354.3147, -3412.591, -3503.6265, -3578.3242, -3632.559, -3674.6567, -3717.5613, -3760.2085, -3788.4807, -3822.624, -3886.9236, -4004.4746, -4150.279, -4275.658, -4371.1343, -4517.594, -4748.125, -5043.127, -5357.7905, -5594.1626, -5728.955, -5791.837, -5821.051, -5945.0234, -6042.6045, -6269.1646, -6444.8164, -6488.951, -6463.709, -6417.9893, -6300.529, -6121.759, -5894.1436, -5658.52, -5456.4688, -5284.7534, -5078.4688, -4691.5474, -4208.8916, -3812.881, -3531.142, -3295.923, -2979.6243, -2702.403, -2465.0664, -2340.9219, -2097.17, -1764.5958, -1465.0876, -1289.7584, -1261.2538, -1155.8845, -875.448, -487.95834, -125.535484, 99.51698, 215.74574, 237.0961, 210.36597, 233.83284, 351.42493, 575.4048, 713.39996, 591.6882, 235.66747, -160.06378, -470.45337, -728.3247, -1130.0037, -1907.9877, -3134.0913, -4806.301, -6591.9385, -8245.65, -9641.673, -11063.912, -12756.747, -14668.968, -16419.807, -17560.588, -17883.531, -17468.006, -16536.309, -15153.72, -13293.506, -10980.588, -8442.008, -5816.3774, -3223.0466, -874.42035, 1148.138, 2812.5488, 4310.4, 5843.4893, 7470.219, 8859.427, 9701.182, 10101.541, 10422.849, 10938.736, 11460.388, 11535.792, 10854.295, 9489.613, 7728.591, 5721.2812, 3433.8037, 929.5206, -1614.9282, -4010.6729, -6161.148, -8064.6255, -9505.111, -10262.015, -10311.397, -9881.438, -9316.553, -8667.981, -7696.0635, -6312.036, -4802.836, -3579.4102, -2798.479, -2168.335, -1299.4071, -188.82898, 817.97046, 1364.2577, 1455.4073, 1340.413, 1172.195, 860.4955, 188.61057, -727.37885, -1466.7306, -1651.6559, -1343.0076, -854.909, -293.11026, 550.18317, 1932.9786, 3756.791, 5543.1865, 6905.615, 7890.103, 8879.846, 10108.995, 11440.332, 12552.374, 13270.207, 13719.488, 14076.94, 14429.957, 14793.695, 15173.486, 15574.112, 15887.052, 15819.524, 15169.148, 14030.438, 12642.83, 11140.199, 9430.6875, 7343.2993, 4976.5376, 2656.0288, 680.6262, -849.7571, -2065.842, -3166.9216, -4218.2095, -5159.5757, -5802.662, -6012.5596, -5725.799, -5153.308, -4663.331, -4593.6265, -4907.403, -5107.019, -4628.931, -3418.3245, -2114.0342, -1617.5759, -2304.073, -3523.4424, -3778.2415, -1572.5349, 3403.808, 7729.125, -2325.3435, -66549.73, -283151.2, -863889.2, -2211964.2, -5009538.5, -10285766.0, -19424528.0, -34071836.0, -55921188.0, -86389304.0, -126236650.0, -175222030.0, -231894060.0, -293603330.0, -356768600.0, -417360130.0, -471498020.0, -516024420.0, -548913860.0, -569431300.0, -578020000.0, -575977660.0, -565028540.0, -546915900.0, -523109200.0, -494668220.0, -462244860.0, -426166050.0, -386530850.0, -343281280.0, -296242400.0, -245163170.0, -189805420.0, -130108890.0, -66423364.0, 249669.22, 68094360.0, 134483660.0, 196198050.0, 249845330.0, 292400220.0, 321735330.0, 337007260.0, 338798050.0, 328976450.0, 310323840.0, 286026400.0, 259161150.0, 232289400.0, 207226640.0, 185001700.0, 165967630.0, 149997980.0, 136699200.0, 125586270.0, 116197870.0, 108151704.0, 101156130.0, 94997500.0, 89520030.0, 84607340.0, 80170216.0, 76139130.0, 72459610.0, 69088180.0, 65988560.0, 63129212.0, 60482548.0, 58025164.0, 55737748.0, 53604180.0, 51610188.0, 49742596.0, 47989532.0, 46340770.0, 44787724.0, 43322904.0, 41939316.0, 40630410.0, 39390256.0, 38213780.0, 37096572.0, 36034572.0, 35023956.0, 34061160.0, 33142948.0, 32266408.0, 31428874.0, 30627914.0, 29861316.0, 29127082.0, 28423292.0, 27748110.0, 27099794.0, 26476808.0, 25877820.0, 25301634.0, 24747098.0, 24213066.0, 23698396.0, 23201954.0, 22722700.0, 22259874.0, 21812940.0, 21381412.0, 20964506.0, 20561162.0, 20170422.0, 19791798.0, 19425206.0, 19070462.0, 18726940.0, 18393746.0, 18070158.0, 17755864.0, 17450770.0, 17154632.0, 16867020.0, 16587480.0, 16315690.0, 16051376.0, 15794226.0, 15543946.0, 15300332.0, 15063210.0, 14832303.0, 14607269.0, 14387887.0, 14174115.0, 13965851.0, 13762719.0, 13564227.0, 13370166.0, 13180766.0, 12996300.0, 12816595.0, 12641067.0, 12469229.0, 12301076.0, 12136849.0, 11976552.0, 11819833.0, 11666367.0, 11516190.0, 11369511.0, 11226259.0, 11086005.0, 10948349.0, 10813316.0, 10681234.0, 10552244.0, 10426051.0, 10302183.0, 10180400.0, 10060833.0, 9943723.0, 9829127.0, 9716865.0, 9606665.0, 9498353.0, 9391921.0, 9287492.0, 9185194.0, 9085011.0, 8986746.0, 8890148.0, 8795085.0, 8701601.0, 8609789.0, 8519665.0, 8431187.0, 8344325.0, 8259052.0, 8175245.5, 8092701.0, 8011315.0, 7931227.0, 7852678.5, 7775736.0, 7700194.5, 7625790.0, 7552453.5, 7480303.0, 7409418.0, 7339699.0, 7270989.5, 7203266.0, 7136623.5, 7071129.5, 7006730.5, 6943329.0, 6880866.5, 6819325.0, 6758695.0, 6699005.5, 6640337.0, 6582720.5, 6526003.0, 6469918.5, 6414364.5, 6359542.5, 6305744.0, 6252967.5, 6200873.0, 6149155.0, 6097900.5, 6047482.5, 5998098.0, 5949542.0, 5901467.0, 5853782.5, 5806694.5, 5760376.5, 5714761.0, 5669686.5, 5625147.5, 5581288.0, 5538149.5, 5495572.5, 5453379.5, 5411599.0, 5370410.0, 5329891.5, 5289923.5, 5250355.5, 5211203.5, 5172598.5, 5134591.0, 5097086.5, 5059994.5, 5023335.0, 4987181.0, 4951502.5, 4916176.5, 4881143.0, 4846493.5, 4812357.0, 4778743.0, 4745519.0, 4712547.5, 4679812.5, 4647410.0, 4615455.0, 4584020.5, 4553096.0, 4522570.0, 4492263.5, 4462064.0, 4432052.0, 4402462.5, 4373470.5, 4345004.0, 4316814.5, 4288724.0, 4260786.5, 4233176.0, 4205975.5, 4179115.0, 4152508.5, 4126174.8, 4100183.2, 4074523.8, 4049086.8, 4023798.5, 3998717.0, 3973963.8, 3949583.0, 3925509.2, 3901664.0, 3878039.2, 3854664.8, 3831525.2, 3808567.0, 3785792.2, 3763290.2, 3741130.8, 3719249.8, 3697480.5, 3675734.5, 3654115.2, 3632819.2, 3611933.2, 3591345.2, 3570864.8, 3550415.2, 3530078.8, 3509973.5, 3490128.0, 3470494.0, 3451042.2, 3431806.5, 3412810.2, 3394000.0, 3375283.5, 3356636.5, 3338129.8, 3319836.5, 3301734.8, 3283733.2, 3265796.2, 3248017.5, 3230539.8, 3213406.8, 3196505.0, 3179669.5, 3162828.2, 3146058.8, 3129502.5, 3113240.8, 3097231.5, 3081359.0, 3065524.8, 3049717.5, 3034000.5, 3018450.2, 3003100.0, 2987941.5, 2972958.2, 2958132.0, 2943417.8, 2928742.0, 2914063.2, 2899443.8, 2885005.8, 2870793.5, 2856696.2, 2842554.8, 2828356.0, 2814283.8, 2800534.0, 2787097.5, 2773775.0, 2760400.0, 2747008.8, 2733751.8, 2720692.2, 2707758.2, 2694888.8, 2682144.0, 2669601.5, 2657199.0, 2644762.0, 2632229.2, 2619771.5, 2607619.2, 2595785.2, 2584038.0, 2572170.2, 2560247.8, 2548519.0, 2537108.5, 2525864.0, 2514537.8, 2503069.0, 2491636.8, 2480442.0, 2469500.8, 2458666.0, 2447814.2, 2436957.0, 2426178.8, 2415515.5, 2404934.0, 2394411.2, 2383991.0, 2373728.8, 2363597.8, 2353479.8, 2343284.0, 2333069.8, 2323020.0, 2313256.0, 2303667.2, 2293980.2, 2284033.0, 2273984.8, 2264200.5, 2254889.8, 2245874.0, 2236757.5, 2227329.8, 2217758.5, 2208376.0, 2199319.8, 2190426.0, 2181461.5, 2172376.2, 2163307.8, 2154388.2, 2145633.8, 2137009.0, 2128505.8, 2120108.8, 2111726.5, 2103247.0, 2094685.1, 2086210.6, 2077974.8, 2069936.8, 2061907.2, 2053766.6, 2045588.2, 2037523.5, 2029610.5, 2021745.9, 2013840.1, 2005936.8, 1998153.2, 1990534.0, 1983005.0, 1975466.0, 1967891.5, 1960326.0, 1952818.5, 1945386.5, 1938029.5, 1930733.8, 1923459.1, 1916153.8, 1908811.6, 1901505.6, 1894341.0, 1887354.9, 1880476.4, 1873594.1, 1866655.5, 1859702.5, 1852819.4, 1846055.0, 1839388.0, 1832751.0, 1826083.0, 1819371.5, 1812663.9, 1806038.2, 1799548.6, 1793184.4, 1786878.4, 1780558.2, 1774196.0, 1767818.8, 1761478.9, 1755214.4, 1749028.6, 1742897.0, 1736791.4, 1730694.1, 1724599.0, 1718504.6, 1712421.2, 1706386.1, 1700451.9, 1694637.8, 1688885.4, 1683092.2, 1677209.6, 1671301.2, 1665486.4, 1659819.2, 1654245.9, 1648682.5, 1643111.0, 1637566.1, 1632047.2, 1626494.4, 1620877.0, 1615268.0, 1609787.2, 1604473.2, 1599247.5, 1594013.0, 1588756.9, 1583523.6, 1578329.6, 1573152.9, 1568004.5, 1562940.8, 1557965.9, 1552973.9, 1547852.8, 1542649.9, 1537558.5, 1532697.8, 1527955.0, 1523104.0, 1518072.4, 1513020.0, 1508135.5, 1503413.5, 1498694.5, 1493886.2, 1489061.9, 1484324.6, 1479651.2, 1474935.4, 1470147.5, 1465375.4, 1460695.6, 1456079.5, 1451465.2, 1446875.5, 1442386.4, 1437992.2, 1433582.0, 1429080.9, 1424563.9, 1420167.2, 1415909.5, 1411673.5, 1407366.0, 1403033.5, 1398773.9, 1394582.8, 1390361.0, 1386066.5, 1381787.8, 1377630.6, 1373583.9, 1369545.5, 1365462.0, 1361369.5, 1357300.5]

"""







tokenized_input = tokenize_text(text_input, word_index, maxlen=maxlen)


# Predict sentiment
predict_sentiment(model, tokenized_input)






