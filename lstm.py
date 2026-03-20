import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for the gates and cell state
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_C = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, C_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))

        # Forget gate
        f_t = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        # Input gate
        i_t = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        # Candidate cell state
        C_tilde_t = tanh(np.dot(self.W_C, concat) + self.b_C)
        # New cell state
        C_t = f_t * C_prev + i_t * C_tilde_t
        # Output gate
        o_t = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        # New hidden state
        h_t = o_t * tanh(C_t)

        return h_t, C_t

# Example usage
input_size = 1
hidden_size = 50
seq_length = 10

# Initialize LSTM cell
lstm = LSTMCell(input_size, hidden_size)

# Random input sequence of shape (seq_length, input_size)
x_seq = np.random.randn(seq_length, input_size)

# Initialize hidden and cell states
h_prev = np.zeros((hidden_size, 1))
C_prev = np.zeros((hidden_size, 1))

# Unroll the LSTM over the sequence
for t in range(seq_length):
    x_t = x_seq[t, :].reshape(-1, 1)  # Reshape to (input_size, 1)
    h_prev, C_prev = lstm.forward(x_t, h_prev, C_prev)
    # print(h_prev, C_prev)
    print(f"Timestep {t+1}: Hidden state shape = {h_prev.shape, x_t.shape}")

# Final hidden state after processing the entire sequence
print("\nFinal hidden state shape:", h_prev.shape)
# print("\nFinal hidden state shape:", h_prev, C_prev)
