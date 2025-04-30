import torch
import matplotlib.pyplot as plt
from train_ppoM import train_ppo
from train_ppo_crossattn import train_ppo_crossattn

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameter lists
gamma_list = [0.999, 0.9, 0.99] 
mini_batch_list = [32, 64, 128]
gae_lambda_list = [0.8, 0.85, 0.9, 0.99]
clip_ratio_list = [0.2, 0.3, 0.1]
T_list = [2048, 4096, 8192]
entropy_coef_list = [1e-3, 1e-4, 1e-5]
actor_lr_list = [1e-3, 1e-4, 1e-5]
critic_lr_list = [1e-3, 1e-4, 1e-5]

# Plotting function
def plot_results(results, labels, title):
    plt.figure(figsize=(10, 6))
    for result, label in zip(results, labels):
        plt.plot(result, label=label)

    plt.title(f"Effect of {title}")
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend(title=title)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    filename = f"plots/{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

    #plt.show()



# Main experiment loop
def main():
    results = []

    # PPO-attn
    # Test clip ratio
    results = []
    print("Testing different clip ratios...")
    for clip_ratio in clip_ratio_list:
        ep_returns = train_ppo_crossattn(clip_ratio=clip_ratio)
        results.append(ep_returns)
    plot_results(results, [f"clip_ratio={cr}" for cr in clip_ratio_list], "PPOcrossattn Clip Ratio")

    # Test buffer size
    results = []
    print("Testing different buffer sizes...")
    for T in T_list:
        ep_returns = train_ppo_crossattn(T = T)
        results.append(ep_returns)
    plot_results(results, [f"buffer_size={T}" for T in T_list], "PPOcrossattn Buffer Size")
    
    # Test gamma
    print("Testing different gamma values...")
    for gamma in gamma_list:
        ep_returns = train_ppo_crossattn(gamma=gamma)
        results.append(ep_returns)
    plot_results(results, [f"gamma={g}" for g in gamma_list], "PPOcrossattn Gamma")
    
    # Test mini-batch size
    results = []
    print("Testing different mini-batch sizes...")
    for mini_batch in mini_batch_list:
        ep_returns = train_ppo_crossattn(mini_batch_size=mini_batch)
        results.append(ep_returns)
    plot_results(results, [f"mini_batch={mb}" for mb in mini_batch_list], "PPOcrossattn Mini Batch Size")
    
    # Test GAE lambda
    results = []
    print("Testing different GAE lambda values...")
    for gae_lambda in gae_lambda_list:
        ep_returns = train_ppo_crossattn(gae_lambda=gae_lambda)
        results.append(ep_returns)
    plot_results(results, [f"gae_lambda={lam}" for lam in gae_lambda_list], "PPOcrossattn GAE Lambda")

    # Test entropy coeffecient
    results = []
    print("Testing entropy coeffecient...")
    for entropy_coef in entropy_coef_list:
        ep_returns = train_ppo_crossattn(entropy_coef = entropy_coef)
        results.append(ep_returns)
    plot_results(results, [f"entropy_coef={en}" for en in entropy_coef_list], "PPOcrossattn Entropy Coefficient")

    # Test actor_lr
    results = []
    print("Testing actor learning rate...")
    for actor_lr in actor_lr_list:
        ep_returns = train_ppo_crossattn(actor_lr = actor_lr)
        results.append(ep_returns)
    plot_results(results, [f"actor_lr={actlr}" for actlr in actor_lr_list], "PPOcrossattn Actor learning rate")

    # Test critic_lr
    results = []
    print("Testing critic learning rate...")
    for critic_lr in critic_lr_list:
        ep_returns = train_ppo_crossattn(critic_lr = critic_lr)
        results.append(ep_returns)
    plot_results(results, [f"critic_lr={crtlr}" for crtlr in critic_lr_list], "PPOcrossattn Critic learning rate")

    




    # PPO-M
    # Test clip ratio
    results = []
    print("Testing different clip ratios...")
    for clip_ratio in clip_ratio_list:
        ep_returns = train_ppo(clip_ratio=clip_ratio)
        results.append(ep_returns)
    plot_results(results, [f"clip_ratio={cr}" for cr in clip_ratio_list], "PPOM Clip Ratio")

    # Test buffer size
    results = []
    print("Testing different buffer sizes...")
    for T in T_list:
        ep_returns = train_ppo(T = T)
        results.append(ep_returns)
    plot_results(results, [f"buffer_size={T}" for T in T_list], "PPOM Buffer Size")
    
    # Test gamma
    print("Testing different gamma values...")
    for gamma in gamma_list:
        ep_returns = train_ppo(gamma=gamma)
        results.append(ep_returns)
    plot_results(results, [f"gamma={g}" for g in gamma_list], "PPOM Gamma")
    
    # Test mini-batch size
    results = []
    print("Testing different mini-batch sizes...")
    for mini_batch in mini_batch_list:
        ep_returns = train_ppo(mini_batch_size=mini_batch)
        results.append(ep_returns)
    plot_results(results, [f"mini_batch={mb}" for mb in mini_batch_list], "PPOM Mini Batch Size")
    
    # Test GAE lambda
    results = []
    print("Testing different GAE lambda values...")
    for gae_lambda in gae_lambda_list:
        ep_returns = train_ppo(gae_lambda=gae_lambda)
        results.append(ep_returns)
    plot_results(results, [f"gae_lambda={lam}" for lam in gae_lambda_list], "PPOM GAE Lambda")

    # Test entropy coeffecient
    results = []
    print("Testing entropy coeffecient...")
    for entropy_coef in entropy_coef_list:
        ep_returns = train_ppo(entropy_coef = entropy_coef)
        results.append(ep_returns)
    plot_results(results, [f"entropy_coef={en}" for en in entropy_coef_list], "PPOM Entropy Coefficient")

    # Test actor_lr
    results = []
    print("Testing actor learning rate...")
    for actor_lr in actor_lr_list:
        ep_returns = train_ppo(actor_lr = actor_lr)
        results.append(ep_returns)
    plot_results(results, [f"actor_lr={actlr}" for actlr in actor_lr_list], "PPOM Actor learning rate")

    # Test critic_lr
    results = []
    print("Testing critic learning rate...")
    for critic_lr in critic_lr_list:
        ep_returns = train_ppo(critic_lr = critic_lr)
        results.append(ep_returns)
    plot_results(results, [f"critic_lr={crtlr}" for crtlr in critic_lr_list], "PPOM Critic learning rate")

    

# Call main if running directly
if __name__ == "__main__":
    main()
