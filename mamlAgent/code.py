import torch
import torch.nn as nn
import learn2learn as l2l
import torchvision

# 1. Dataset and Task Distribution
omniglot_dataset = torchvision.datasets.Omniglot(root='./data', download=True, transform=torchvision.transforms.ToTensor())
omniglot = l2l.data.MetaDataset(omniglot_dataset)
#omniglot = l2l.data.MetaDataset(l2l.vision.datasets.Omniglot(root='./data', transform=torchvision.transforms.ToTensor()))
taskset = l2l.data.TaskDataset(omniglot, num_tasks=20, task_transforms=[
                                l2l.data.transforms.NWays(omniglot, n=5),
                                l2l.data.transforms.KShots(omniglot, k=1),
                                l2l.data.transforms.LoadData(omniglot),
                                l2l.data.transforms.RemapLabels(omniglot),
                                l2l.data.transforms.ConsecutiveLabels(omniglot)
                            ])

# 2. Define the Model
class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleMLP, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.main(x)

# 3. Meta-SGD Adaptation
def meta_sgd_adaptation(model, lr_params, loss, step_size=0.5):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    for (name, param), lr, g in zip(model.named_parameters(), lr_params, grads):
        new_val = param - lr * g
        setattr(model, name, new_val)

# 4. Training Loop
in_dim = 28 * 28  # Assuming flattened Omniglot images
out_dim = 5  # 5-way classification
model = SimpleMLP(in_dim, out_dim)
meta_lr = 0.001
step_size = 0.5
num_iterations = 1000
num_tasks = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=meta_lr)
lr_params = [torch.nn.Parameter(step_size * torch.ones_like(p)) for p in model.parameters()]

for iteration in range(num_iterations):
    optimizer.zero_grad()
    meta_train_loss = 0.0

    for task in range(num_tasks):
        learner = l2l.clone_module(model)

        # Sample data
        data, labels = taskset.sample()
        train_data, train_labels = data[0], labels[0]
        val_data, val_labels = data[1], labels[1]

        # Inner loop
        predictions = learner(train_data)
        loss = criterion(predictions, train_labels)
        meta_sgd_adaptation(learner, lr_params, loss)

        # Outer loop
        predictions = learner(val_data)
        meta_train_loss += criterion(predictions, val_labels)

    # Update main model
    meta_train_loss.backward()
    optimizer.step()
