import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Data Generation
def generate_data(n_points=1000):
    # Generate x values from -2*pi to 2*pi
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n_points).view(-1, 1)
    # Generate corresponding y values (sine function)
    y = torch.sin(x)
    # y = torch.cos(x)
    return x, y

# 2. Neural Network Architecture
class SineApproximator(nn.Module):
    def __init__(self):
        super(SineApproximator, self).__init__()
        # A simple feed-forward network
        # Input layer (1) -> Hidden Layer 1 (64) -> Hidden Layer 2 (64) -> Output (1)
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),            # Tanh is better for smooth functions like sine
            # nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)      # Final output layer
        )

    def forward(self, x):
        return self.network(x)

# 3. Training Setup
def train_model():
    # Hyperparameters
    epochs = 2000
    learning_rate = 0.01
    
    # Prepare data
    x_train, y_train = generate_data(1000)

    # Initialize Model, Loss Function, and Optimizer
    model = SineApproximator()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    
    # Training Loop
    for epoch in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_train)

        # Compute loss
        loss = criterion(y_pred, y_train)

        # Backward pass: Zero gradients, perform backpropagation, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress every 200 epochs
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    return model, x_train, y_train

# 4. Execution and Visualization
if __name__ == "__main__":
    # Train the model
    model, x_train, y_train = train_model()

    # Set model to evaluation mode
    model.eval()

    # Generate predictions for plotting
    with torch.no_grad():
        y_predicted = model(x_train)
        # y_predicted = model(torch.linspace(2 * np.pi, 4 * np.pi, 1000).view(-1, 1))

    # Plotting the results
    plt.figure(figsize=(10, 6))
    # plt.plot(torch.linspace(2 * np.pi, 4 * np.pi, 1000).view(-1, 1).numpy(), y_predicted.numpy(), label='extended Sine Wave', color='magenta', linewidth=2)
    plt.plot(x_train.numpy(), y_train.numpy(), label='True Sine Wave', color='blue', linewidth=2)
    plt.plot(x_train.numpy(), y_predicted.numpy(), label='NN Approximation', color='red', linestyle='--')
    plt.title('Neural Network Approximation of $\sin(x)$')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
