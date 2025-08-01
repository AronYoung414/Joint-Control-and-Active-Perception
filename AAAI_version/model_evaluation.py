import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from random import choices
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import your POMDP environment
from product_pomdp import prod_pomdp

# Import the PolicyNetwork and helper functions from your training code
# Assuming your training code is saved as 'policy_gradient_training.py'
from observable_opeator_policy_network import *


# Initialize the POMDP
# prod_pomdp = prod_pomdp()


def get_action_probabilities(policy_net, obs_sequence):
    """
    Get action probabilities from the policy network
    """
    obs_indices = [obs_to_index(obs) for obs in obs_sequence]
    if len(obs_indices) == 0:
        obs_tensor = torch.zeros((1, 1), dtype=torch.long)
    else:
        obs_tensor = torch.tensor([obs_indices], dtype=torch.long)

    with torch.no_grad():
        action_probs, _ = policy_net(obs_tensor)
        return action_probs[0].numpy()


def generate_single_trajectory(policy_net, T, observable_operator, initial_state_type='mix', verbose=False):
    """
    Generate a single trajectory using the trained policy

    Args:
        policy_net: Trained policy network
        T: Maximum trajectory length
        initial_state_type: 'mix', '0', or '1' for initial state selection
        verbose: Whether to print step-by-step information

    Returns:
        trajectory: Dict containing states, observations, actions, and metadata
    """
    states = []
    observations = []
    actions = []
    a_list = []
    action_probs_history = []
    posterior_adversarys = []
    posterior_normals = []

    # Start from initial state
    if initial_state_type == 'mix':
        state = choices(prod_pomdp.initial_states, prod_pomdp.initial_dist_sampling, k=1)[0]
    elif initial_state_type == '1':
        state = prod_pomdp.initial_states[1]
    elif initial_state_type == '0':
        state = prod_pomdp.initial_states[0]
    else:
        raise ValueError('Invalid initial_state_type. Use "mix", "0", or "1".')

    states.append(state)

    if verbose:
        print(f"Starting state: {state}")

    # Sample first sensing action
    act = action_sampler_network(policy_net, [], is_last_step=False)
    actions.append(act)
    a = prod_pomdp.actions.index(act)
    a_list.append(a)
    action_probs = get_action_probabilities(policy_net, [])
    action_probs_history.append(action_probs.copy())

    # Get the observation of initial state
    obs = prod_pomdp.observation_function_sampler(state, act)
    observations.append(obs)

    # Get the prior distribution of secrets
    posterior_adversary = 0.5
    posterior_normal = 1 - posterior_adversary
    posterior_adversarys.append(posterior_adversary)
    posterior_normals.append(posterior_normal)

    if verbose:
        print(f"Step 0: Action={act}, Observation={obs}")
        print(f"Action probabilities: {dict(zip(prod_pomdp.actions[:-1], action_probs))}")

    # Generate trajectory for T-1 steps
    for t in range(T - 1):
        # Sample the next state
        state = prod_pomdp.next_state_sampler(state, act)
        states.append(state)

        # Sample sensing action using observation history
        act = action_sampler_network(policy_net, observations, is_last_step=False)
        actions.append(act)
        a = prod_pomdp.actions.index(act)
        a_list.append(a)
        action_probs = get_action_probabilities(policy_net, observations)
        action_probs_history.append(action_probs.copy())

        # Add the observation
        obs = prod_pomdp.observation_function_sampler(state, act)
        observations.append(obs)

        # Calculate the posterior
        posterior_adversary, posterior_normal = posterior(observations, a_list, observable_operator)
        posterior_adversarys.append(posterior_adversary)
        posterior_normals.append(posterior_normal)

        if verbose:
            print(f"Step {t + 1}: State={state}, Action={act}, Observation={obs}")
            print(f"Action probabilities: {dict(zip(prod_pomdp.actions[:-1], action_probs))}")

    # The last action must be 'end' action
    act = action_sampler_network(policy_net, observations, is_last_step=True)
    actions.append(act)
    a = prod_pomdp.actions.index(act)
    a_list.append(a)

    # Final step: transition to final state
    state = prod_pomdp.next_state_sampler(state, act)
    states.append(state)

    # Get final observation
    obs = prod_pomdp.observation_function_sampler(state, act)
    observations.append(obs)

    # Calculate the posterior
    posterior_adversary, posterior_normal = p_zT_g_y(observations, a_list, observable_operator)[0:-1]
    posterior_adversarys.append(posterior_adversary)
    posterior_normals.append(posterior_normal)

    if verbose:
        print(f"Final step: State={state}, Action={act}, Observation={obs}")

    # Determine if trajectory is successful (reaches goal states)
    is_successful = state in prod_pomdp.goal_states
    is_adversarial = state in prod_pomdp.secret_states

    trajectory = {
        'states': states,
        'observations': observations,
        'actions': actions,
        'action_probs_history': action_probs_history,
        'posterior_adversary': posterior_adversarys,
        'posterior_normal': posterior_normals,
        'is_successful': is_successful,
        'is_adversarial': is_adversarial,
        'final_state': state,
        'trajectory_length': len(observations)
    }

    return trajectory


def generate_multiple_trajectories(policy_net, num_trajectories, T, observable_operator, initial_state_type='mix'):
    """
    Generate multiple trajectories
    """
    trajectories = []

    print(f"Generating {num_trajectories} trajectories...")

    for i in range(num_trajectories):
        trajectory = generate_single_trajectory(policy_net, T, observable_operator, initial_state_type)
        trajectories.append(trajectory)

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1} trajectories...")

    print("Trajectory generation complete.")

    return trajectories


def plot_confusion_matrix(trajectories, threshold=0.5, save_path=None):
    """
    Create and plot confusion matrix for agent type identification

    Args:
        trajectories: List of trajectory dictionaries
        threshold: Decision threshold for adversary classification (default: 0.5)
        save_path: Optional path to save the plot
    """
    # Extract true labels and final posterior probabilities
    true_labels = []
    predicted_probs = []

    for traj in trajectories:
        # True label: 1 if adversarial, 0 if normal
        true_labels.append(int(traj['is_adversarial']))

        # Final posterior probability for adversary
        final_posterior_adversary = traj['posterior_adversary'][-1]
        predicted_probs.append(final_posterior_adversary)

    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    # Apply threshold to get predicted labels
    predicted_labels = (predicted_probs >= threshold).astype(int)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Set font sizes
    plt.rcParams.update({'font.size': 24})

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Adversary'],
                yticklabels=['Normal', 'Adversary'],
                annot_kws={'size': 32})
    plt.title(f'Confusion Matrix for Agent Type Identification\n(Threshold = {threshold})', fontsize=26)
    plt.xlabel('Predicted Label', fontsize=32)
    plt.ylabel('True Label', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels,
                                target_names=['Normal', 'Adversary']))

    # Calculate and print accuracy metrics
    accuracy = np.mean(predicted_labels == true_labels)
    precision_adversary = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    recall_adversary = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    print(f"\nOverall Accuracy: {accuracy:.3f}")
    print(f"Adversary Detection Precision: {precision_adversary:.3f}")
    print(f"Adversary Detection Recall: {recall_adversary:.3f}")

    return cm, true_labels, predicted_probs


def display_sample_trajectories(trajectories, num_to_show=5):
    """
    Display detailed information for sample trajectories
    """
    print(f"\n{'=' * 50}")
    print(f"SAMPLE TRAJECTORIES (showing {num_to_show})")
    print(f"{'=' * 50}")

    for i, traj in enumerate(trajectories[:num_to_show]):
        print(f"\n--- Trajectory {i + 1} ---")
        print(f"Final state: {traj['final_state']}")
        print(f"Successful: {traj['is_successful']}")
        print(f"Adversarial: {traj['is_adversarial']}")
        print(f"Length: {traj['trajectory_length']}")

        print("Step-by-step:")
        for t in range(len(traj['observations'])):
            if t < len(traj['actions']):
                print(f"  Step {t}: State={traj['states'][t]}, "
                      f"Action={traj['actions'][t]}, Observation={traj['observations'][t]}, "
                      f"Posterior=({traj['posterior_normal'][t]}, {traj['posterior_adversary'][t]})")

        print(f"Final state: {traj['states'][-1]}")
        print("-" * 30)


def load_trained_policy(model_path, obs_vocab_size, action_size, hidden_dim=64, max_seq_len=20):
    """
    Load the trained policy network from file
    """
    # Initialize the network
    policy_net = PolicyNetwork(obs_vocab_size, action_size, hidden_dim, max_seq_len)

    # Load the trained weights
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location='cpu')

    policy_net.load_state_dict(state_dict)
    policy_net.eval()  # Set to evaluation mode

    print(f"Loaded trained policy from {model_path}")
    return policy_net


def main():
    # Configuration
    ex_num = 20  # Should match the experiment number from your training
    model_path = f'./data/Values/policy_net_{ex_num}.pkl'

    # Network parameters (should match your training configuration)
    obs_vocab_size = len(prod_pomdp.observations)
    action_size = prod_pomdp.action_size
    hidden_dim = 64
    max_seq_len = 8

    # Trajectory generation parameters
    num_trajectories = 1000
    T = 8  # Maximum trajectory length
    initial_state_type = 'mix'  # Can be 'mix', '0', or '1'

    try:
        # Load the trained policy
        policy_net = load_trained_policy(model_path, obs_vocab_size, action_size, hidden_dim, max_seq_len)

        # Use lazy computation for maximum efficiency with large state spaces
        print("Initializing lazy observable operator for efficient computation...")
        obs_dict = get_observable_operator_ultra_fast()  # Uses LazyObservableOperator

        # Generate trajectories
        trajectories = generate_multiple_trajectories(policy_net, num_trajectories, T, obs_dict, initial_state_type)

        # Display sample trajectories
        display_sample_trajectories(trajectories, num_to_show=5)  # Reduced to 5 for brevity

        # NEW: Create and display confusion matrix
        print(f"\n{'=' * 50}")
        print("CONFUSION MATRIX ANALYSIS")
        print(f"{'=' * 50}")

        cm, true_labels, predicted_probs = plot_confusion_matrix(
            trajectories,
            threshold=0.5,
            save_path=f'./data/confusion_matrix_{ex_num}.pdf'
        )

        # Save results if desired
        save_results = True
        if save_results:
            import pickle
            results = {
                'trajectories': trajectories,
                'confusion_matrix': cm,
                'true_labels': true_labels,
                'predicted_probs': predicted_probs
            }
            with open(f'./data/generated_trajectories_{ex_num}.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"\nResults saved to ./data/generated_trajectories_{ex_num}.pkl")

        return trajectories

    except FileNotFoundError:
        print(f"Error: Could not find trained policy at {model_path}")
        print("Make sure you have run the training script and the file exists.")
    except Exception as e:
        print(f"Error loading or using the trained policy: {e}")


if __name__ == "__main__":
    main()