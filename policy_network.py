import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from product_pomdp import prod_pomdp
# prod_pomdp = prod_pomdp()


# Define a deeper policy network with two hidden layers
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # Output layer
        self.activation = nn.ReLU()  # Activation function

    def stable_softmax(self, logits):
        max_logits = logits.max(dim=-1, keepdim=True)[0]  # Get the max value per row
        shifted_logits = logits - max_logits  # Shift for numerical stability
        return F.softmax(shifted_logits, dim=-1)  # Apply softmax safely

    def forward(self, state):
        x = self.activation(self.fc1(state))  # Pass through first hidden layer
        x = self.activation(self.fc2(x))  # Pass through second hidden layer
        logits = self.fc3(x)  # Compute logits
        logits = torch.clamp(logits, min=-50, max=50)
        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print("Warning: Inf or NaN detected in logits!")
        probs = self.stable_softmax(logits)  # Convert to probability distribution
        # we can add exp() to avoid numerical issue
        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print("Warning: Inf or NaN detected in logits!")
        return probs


def get_action_probability(policy_net, state, action):
    """
    Computes the probability of taking a specific action given a state.

    Args:
        policy_net (nn.Module): The policy network.
        state (torch.Tensor): The state tensor (shape: [1, state_dim]).
        action (int): The chosen action.

    Returns:
        float: Probability of the chosen action.
    """
    state = torch.tensor([[state]], dtype=torch.float32)
    with torch.no_grad():  # No need to compute gradients
        probs = policy_net(state)  # Compute action probabilities
        action_prob = probs[0, action].item()  # Extract probability of selected action

    return action_prob


# Function to compute a single flattened gradient vector of log policy
def compute_log_policy_gradient(policy_net, state, action):
    state = torch.tensor([[state]], dtype=torch.float32)
    policy_net.zero_grad()  # Clear previous gradients
    probs = policy_net(state)  # Compute action probabilities
    log_probs = torch.log(probs)  # Compute log probabilities

    # Compute ∇θ log πθ(a | s)
    log_prob_action = log_probs[0, action]  # Get log πθ(a | s) for chosen action
    log_prob_action.backward()  # Compute gradients

    # Concatenate all gradients into a single vector
    gradients = []
    for param in policy_net.parameters():
        if param.grad is not None:
            gradients.append(param.grad)
    #
    # # Concatenate and convert to a NumPy array
    # gradient_array = torch.cat(gradients).cpu().numpy()
    return gradients


def create_gradient_shaped_tensors(policy_net):
    """
    Creates tensors with the same shape as the gradients of the policy network parameters.
    Args:
        policy_net (nn.Module): The policy network.
    Returns:
        list of torch.Tensor: A list of tensors matching gradient shapes.
    """
    grad_tensors = []
    for param in policy_net.parameters():
        grad_tensors.append(torch.zeros_like(param))  # Tensor of zeros
    return grad_tensors


# Example Usage
# if __name__ == "__main__":
#     # Example usage
#     state_dim = 1  # Example state size
#     action_dim = 3  # Example number of actions
#     hidden_dim = 64  # Hidden layer size
#
#     policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
#     # optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
#
#     state = 2  # Example state
#     action = 1  # Example action taken
#
#     # Zero gradients before backward pass
#     policy_net.zero_grad()
#
#     # Compute single log policy gradient vector
#     gradients = compute_log_policy_gradient(policy_net, state, action)
#     empty_grads = create_gradient_shaped_tensors(policy_net)
#     # for param in policy_net.parameters():
#     #     if param.grad is not None:  # Ensure the gradient exists
#     #         print(param)  # Tensor of zeros
#
#     for param in policy_net.parameters():
#         print("The parameters", param.size())
#
#     # Print the gradient vector
#     for grad in gradients:
#         print("Gradient of log policy (flattened vector):\n", grad.size())
