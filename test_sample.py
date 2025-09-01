import torch
from torch.distributions import Categorical

# Initialize with probabilities
probs = torch.tensor([0.00, 0.1, 0.1, 0.2,0.6,0.00, 0.0])
categorical_dist_probs = Categorical(probs=probs)

# Sample from the distribution
sample_probs = categorical_dist_probs.sample()
print(f"Sample from probs: {sample_probs.item()}")

# Initialize with logits
logits = torch.tensor([1.0, 2.0, 0.5])
categorical_dist_logits = Categorical(logits=logits)

# Sample from the distribution
sample_logits = categorical_dist_logits.sample()
print(f"Sample from logits: {sample_logits.item()}")

# Get log probability
log_prob_value = categorical_dist_probs.log_prob(torch.tensor(1))
print(f"Log probability of category 1: {log_prob_value.item()}")