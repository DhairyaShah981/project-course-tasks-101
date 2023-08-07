import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from create_dataset import create_circle_dataset, create_exclusive_or_dataset, create_gaussian_dataset, create_spiral_dataset, create_moon_dataset
from basis_functions import sine_basis, gaussian_basis, polynomial_basis


# Set up the Streamlit app layout
st.set_page_config(page_title='Neural Network Playground with Uncertainty Visualization')
st.title('Neural Network Playground with Uncertainty Visualization')
st.write(
    '''Create a streamlit app for neural network having following features:\n
    - It should have various controls such as number of neurons in each layer, lr, epochs etc.
    - Show the fit on various datasets.
    - Show the probabilities on the contour plot and not the discrete predictions.
    - Have an option of including various transformations to the inputs (basis functions): Sine basis, Gaussian basis, Polynomial basis and show the effect of each on the fit via the same contour plots.
    - Have an option of enabling MC dropout method for getting uncertainty.
    '''
) 

st.write("---")

# Define the multi-layer perceptron model with MC Dropout
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.2):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_prob))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

def train_MLP(model, train_loader, epochs, criterion, optimizer):
    '''
    Train the MLP model

    Parameters:
        model (torch.nn.Module): The MLP model
        train_loader (torch.utils.data.DataLoader): DataLoader for batch training
        epochs (int): Number of epochs
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim): Optimizer
    
    Returns: None
    '''
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(X_train):.4f}')

def make_predictions(model, x_val, is_MC_dropout, n_samples):
    '''
    Make predictions using the MLP model

    Parameters:
        model (torch.nn.Module): The MLP model
        x_val (torch.tensor): Validation data
        is_MC_dropout (str): Whether MC Dropout is enabled or not
        n_samples (int): Number of samples for MC Dropout
    
    Returns: 
        if MC dropout is Disabled: model output without dropout
        if MC dropout is Enabled: Average output of the predictions with droput
    
    '''
    if is_MC_dropout == "Disabled":
        model.eval()        # this disables dropout
        with torch.no_grad():
            outputs = model(x_val)
            # Reshape the model output to match the target size
            outputs = outputs.view(-1, 1)
            return outputs.squeeze()
    else:
        model.train()    # this enables dropout
        with torch.no_grad():
            prob_sum = torch.zeros(len(x_val))
            for _ in range(n_samples):
                outputs = model(x_val)
                # Reshape the model output to match the target size
                outputs = outputs.view(-1, 1)
                prob_sum += outputs.squeeze()
            avg_prob = prob_sum / n_samples
            return avg_prob

def plot_input_data(X, y, dataset_name):
    '''
    Plot the input data

    Parameters:
        X (torch.tensor): Input data
        y (torch.tensor): Target data
        dataset_name (str): Name of the dataset

    Returns: None
    '''
    st.title("Input Data")
    st.write("Dataset Used:", dataset_name)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, colorscale='viridis', line_width=1)))
    fig.update_layout(xaxis_title='Feature 1', yaxis_title='Feature 2', width=600, height=600)
    st.plotly_chart(fig)

def plot_contour(X_val, avg_prob, dataset_name, basis_function="No Transformation", is_MC_dropout="Disabled"):
    '''
    Plotting the contour plots with probabilities

    Parameters:
        X_val (torch.tensor): Validation data
        avg_prob (torch.tensor): Average probability of the model
        dataset_name (str): Name of the dataset
        basis_function (str): Name of the basis function to transform the input data
        is_MC_dropout (str): Whether MC Dropout is enabled or not
    
    Returns: None
    '''
    st.write("---")
    st.title("Probability Contour Plot")
    st.write("Dataset Used:", dataset_name)
    st.write("Basis Function Used:", basis_function)
    st.write("MC Dropout:", is_MC_dropout)
    plt.figure()
    x_min, x_max = X_val[:, 0].min() - 0.1, X_val[:, 0].max() + 0.1
    y_min, y_max = X_val[:, 1].min() - 0.1, X_val[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100), torch.linspace(y_min, y_max, 100))
    xx_axis=xx
    yy_axis=yy
    xx=xx.reshape(-1, 1)
    yy=yy.reshape(-1, 1)

    if basis_function == "No Transformation":
        xx=xx
        yy=yy
        X_grid = torch.cat((xx, yy), dim=1)
    elif basis_function == "Sine Basis":
        xx=sine_basis(xx)
        yy=sine_basis(yy)
        X_grid = torch.cat((xx, yy), dim=1)
    elif basis_function == "Gaussian Basis":
        xx=gaussian_basis(xx).squeeze()
        yy=gaussian_basis(yy).squeeze()
        X_grid = torch.cat((xx, yy), dim=1)
    elif basis_function == "Polynomial Basis":
        X_grid = torch.cat((xx, yy), dim=1)
        X_grid=polynomial_basis(X_grid)
    else:
        st.write("Invalid Basis Function Selection.")

    probs = make_predictions(model, X_grid, is_MC_dropout, mc_dropout_samples)
    Z = probs.reshape(xx_axis.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx_axis, yy_axis, Z, levels=20, cmap="RdBu", alpha=0.8)
    plt.colorbar(label="Probability")
    plt.scatter(X_val[:, 0], X_val[:, 1], c=avg_prob, cmap="viridis", edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    fig = plt.gcf()
    st.pyplot(fig)

def prepare_data_and_loaders(X, y, train_ratio):
    '''
    Make train and validation data and create DataLoader for batch training

    parameters:
        X (torch.tensor): Input data
        y (torch.tensor): Target data
        train_ratio (float): Ratio of training data to total data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for batch training
        X_train (torch.tensor): Training data
        X_val (torch.tensor): Validation data
        y_train (torch.tensor): Training target
        y_val (torch.tensor): Validation target
    '''
    X = X.view(X.shape[0], -1)
    num_train = int(train_ratio * len(X))
    X_train = X[:num_train]
    y_train = y[:num_train]
    X_val = X[num_train:]
    y_val = y[num_train:]
    # Create DataLoader for batch training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    return train_loader, X_train, X_val, y_train, y_val

# Button used to train the model
train_model = st.sidebar.button("Train Model")

# Take inputs from the streamlit
dataset_name = st.sidebar.selectbox("Select Dataset:", ("Circle Dataset", "XOR Dataset", "Gaussian Dataset", "Spiral Dataset", "Moon Dataset"))
basis_function = st.sidebar.selectbox("Input Transform Basis Function:", ("No Transformation", "Sine Basis", "Gaussian Basis", "Polynomial Basis"))

num_datapoints = st.sidebar.slider("Number of Datapoints", min_value=100, max_value=5000, value=500)
train_ratio = st.sidebar.slider("Training set ratio", min_value=0.6, max_value=0.95, value=0.8)

# Generating dataset based on the selection
if dataset_name == "Circle Dataset":
    X, y = create_circle_dataset(n_samples=num_datapoints)
elif dataset_name == "XOR Dataset":
    X, y = create_exclusive_or_dataset(n_samples=num_datapoints)
elif dataset_name == "Gaussian Dataset":
    X, y = create_gaussian_dataset(n_samples=num_datapoints)
elif dataset_name == "Spiral Dataset":
    X, y = create_spiral_dataset(n_samples=num_datapoints)
elif dataset_name == "Moon Dataset":
    X, y = create_moon_dataset(n_samples=num_datapoints)
else:
    st.write("Invalid Dataset Selection.")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Transform the input data based on the selection
if basis_function == "No Transformation":
    train_loader, X_train, X_val, y_train, y_val = prepare_data_and_loaders(X, y, train_ratio)
elif basis_function == "Sine Basis":
    train_loader, X_train, X_val, y_train, y_val = prepare_data_and_loaders(sine_basis(X), y, train_ratio)
elif basis_function == "Gaussian Basis":
    train_loader, X_train, X_val, y_train, y_val = prepare_data_and_loaders(gaussian_basis(X), y, train_ratio)
elif basis_function == "Polynomial Basis":
    train_loader, X_train, X_val, y_train, y_val = prepare_data_and_loaders(polynomial_basis(X), y, train_ratio)
else:
    st.write("Invalid Basis Function Selection.")

learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01)
epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=5000, value=500)
st.sidebar.write("---")
is_MC_dropout = st.sidebar.selectbox("MC Dropout", ("Disabled", "Enabled"))
dropout_prob = st.sidebar.slider("Architecture Drop-out Probability", min_value=0.1, max_value=0.8, value=0.2)

if is_MC_dropout == "Enabled":
    mc_dropout_samples = st.sidebar.slider("MC Droput Samples", min_value=1, max_value=20, value=5)
else:
    mc_dropout_samples = 1

st.sidebar.write("---")

# Take input for the number of hidden layers using a slider
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=10, value=2)

# Create a list to store the number of neurons for each hidden layer
hidden_layer_neurons = []

# Take input for the number of neurons in each hidden layer using sliders
for i in range(1, num_hidden_layers + 1):
    num_neurons = st.sidebar.slider(f"Number of Neurons in Hidden Layer {i}", min_value=1, max_value=100, value=10)
    hidden_layer_neurons.append(num_neurons)

# Initialize the models and other hyperparameters
input_size = X_train.shape[1]
output_size=1   # Binary classification

# Train the model if the button is pressed
if train_model:
    train_model = False
    plot_input_data(X, y, dataset_name)
    model = MLP(input_size, hidden_layer_neurons, output_size, dropout_prob=dropout_prob)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_MLP(model, train_loader, epochs, criterion, optimizer)
    avg_prob = make_predictions(model, X_val, is_MC_dropout, mc_dropout_samples)
    plot_contour(X_val, avg_prob, dataset_name, basis_function, is_MC_dropout)
