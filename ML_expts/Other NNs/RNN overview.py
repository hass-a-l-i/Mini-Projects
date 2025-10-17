# RNN remembers past inputs - not feed forward as has loops and state of each neuron can be seen as its memory
# output dependent on all elements before it - not independent each layer like normal NNs
# means same weights/biases in each layer instead of different ones  - using grad descent these are adjusted to min loss
# output of each layer dep on input => each layer has a hidden layers which are the NN and it loos through this many times
# RNNs use backpropagation through time to find the gradient - i.e. we find gradient each layer starting with final layer then go back to first, so we go backward through NN because the gradient of current layer when doing chain rule includes partial deriv related to prev layer
# BPTT sums error each t step as same params shared throughout RNN
# simple RNN has single tanh activation in each layer which uses input to current layer and input from prev layer to produce output
# can have any number of output and input combos with RNNs - 4 types are 1:1, 1:many, many:1 and many:many
# unlike single input/output (for image classification) conventional feedforward CNNs, RNNs applicable to time series analysis, text generation etc...

# problems with RNNs are vanishing gradient when grad become small and params stop updating so learning stops OR  exploding gradient where gradient too large so model unstable and params too large = longer training and poor performance
# solve this by using different RNN structures e.g. long short term memory (LSTM) and gated recurrent units (GRU) - allows for long period of info to be remembered by layers

# LSTM has multiple activation layers within each layer - uses input gate (how much current input let through), output gate (how much output state you want to expose the next part of network) and forget gate (how much of previous layer memory you want to let through)
# also uses previous state and current state of current layer combined with above to compute output (which is hidden state of layer but then this x by output gate, so we decide how much of this state relevant to be shared with rest of NN)
# basically allows to customize dependencies for current layer

# GRU similar to LSTM, but no 3 gates, now we have 2 gates => uses reset and update gate to solve vanishing gradient
# reset gate determines how combine new input with memory (state) of last layer and update gate determines how much of prev state we keep in current state
# no internal memory (current state) which is separate from output (hidden state) and ofc no ouput gate to control outflow of info
# instead input and forget gates coupled to become update gate and reset gate directly applied to previous hidden state#
# also we only use activation function once for non-linearity but in LSTM we used it twice, once to find combo of 3 gates output which is used to find current state and then again on current state to find hidden state output
# quicker training with GRU as simpler


