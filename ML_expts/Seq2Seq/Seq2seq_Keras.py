# seq 2 seq trains models to convert sequences from one domain to another e.g. sentences from english to french
# generally - machine language gives input and output different lengths
# steps:
# 1. RNN layer acts as encoder processing input to get own internal state - only care about state not output of RNN (is the context that the decoder uses in next step)
# 2. another RNN is decoder layer, predicting next chars of sequence given previous chars of output sequence (called target sequence) => output of last t step is fed in as input in the next time step
# train this second RNN to specifically produce target sequence but offset one t step in future (called teacher forcing)
# encoder learns to generate t+1 target given t target, conditional on input sequence
# Uses START and END tokens to mark beginning and end of sentence so algo knows where to start seq from and where to end
# if instead want to decode unknown inputs (e.g. translate from foreign to native) we instead:
# 1. encode input sequence into state vectors
# 2. start with target sequence size 1 e.g. one char
# 3. use state vec of input and 1 char target sequence and use decoder to predict next char
# 4. use argmax to sample next char using the prediction we made
# 5. append new char to target sequence
# 6. repeat until we reach END token
# builds sentence as it goes using the previous state as input (Mary, Mary had, Mary had a ....)

# now we create model to translate short ENG sentenves into FRE using chars (models usually use word by word translation instead)

# architecture:
# start with set inputs and targets from our chosen domains (ENG and FRE)
# encoder LSTM turns input seq into 2 state vecs (keep LSTM state and discard outputs remember
# decoder LSTM trained to turn target into same sequence but offset one t-step in future - teacher forcing (decoder learns to generate t+1 target given)
# use method above for translating unknown inputs - use current char and argmax to produce predictions for next char using argmax

# code:
import numpy as np
import keras
import os
from pathlib import Path

# configure all vars
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space - helps define shape of tensors
num_samples = 10000  # Number of samples to train on.
data_path = "fra.txt"  # Path to the data txt file on disk.

# turn data into vectors
# deconstruct texts into sets of chars
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")  # import texts in python
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"  # create sentence with start and end tokens
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)  # list of all unique chars for input
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)  # same for target

input_characters = sorted(list(input_characters))  # these are our encoder tokens
target_characters = sorted(list(target_characters))  # decoder tokens
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

# make dictionary of chars
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# make blank arrays of 0s for our data shape
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype="float32",
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # encoder array filled with data (1s and 0s), 1s for chars, 0s for spaces
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character to ensure no crashes.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# now build model
# Define LSTM encoder. encoder inputs in ML feedable shape and encode the inputs to make outputs.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state as we build using hidden state
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences, and to return internal states as well. We don't use the return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)  # define LSTM layer
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)  # decode
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")  # add dense layer
decoder_outputs = decoder_dense(decoder_outputs)  # apply to outputs

# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
"""
# now train model
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)  # rmsprop is variation of gradient descent, creates a moving average of the learning rate to avoid gradient vanishing/blowing up => uses moving av of learning rate as learning progresses
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2s_model.keras")
"""

# now load model - so we can use trained model to make our encoder and decoder for testing
model_loaded = keras.models.load_model("s2s_model.keras")

# below we define the RNNs for encoder and decoder
encoder_inputs = model_loaded.input[0]  # input_1 for encoder
encoder_outputs2, state_h_enc, state_c_enc = model_loaded.layers[2].output  # lstm_1 - returns hidden encoded states
encoder_states = [state_h_enc, state_c_enc]  # make a list out of hidden states
encoder_model = keras.Model(encoder_inputs, encoder_states)  # make encoder model

decoder_inputs = model_loaded.input[1]  # input_2 for decoder
decoder_state_input_h = keras.Input(shape=(latent_dim,))   # preparing shape of decoder hidden states
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]  # same format as encoder
decoder_lstm = model_loaded.layers[3]  # define lstm from saved model layer 4
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(  # same as encode, find hidden states of decoder then store in list
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model_loaded.layers[4]  # import dense layer from loaded model
decoder_outputs = decoder_dense(decoder_outputs)  # apply to decoder outputs to use in decoder model
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states  # load inputs and outputs for training decoder, separate out inputs as hidden states as tuple list + outputs from decoder
)

# Reverse-lookup token index to decode sequences back to something readable - as we have sequence of tokens which we want to shift back to sentences
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# decoder function
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character (which we defined as tab)
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Sample a token and add to output sentence which has been decoded
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1) - want to refresh each loop to ensure it doesn't grow to large - fixes each output to be one char at a time
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0  # repop first char as the start token again to repeat

        # Update states
        states_value = [h, c]  # hidden states updated as new state values for inputs for next loop
    return decoded_sentence

# now can generate decoded sentences
for seq_index in range(20):
    # Take one sequence (part of the training set) for decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]  # returns a word for us to decode as sentence (as we did char by char)
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)


