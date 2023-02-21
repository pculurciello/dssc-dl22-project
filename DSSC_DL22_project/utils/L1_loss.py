import torch

class L1_loss(torch.nn.Module):
    def __init__(self, L1_lambda, loss_fn):
        super().__init__()
        self.L1_lambda = L1_lambda
        self.loss_fn = loss_fn

    def forward(self, y_hat, y, model_parameters):
        # Compute L1 loss component
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model_parameters:
            if 'weight' or 'bias' in name:
                L1_reg = L1_reg + torch.norm(param, 1)

        # Add L1 loss component
        loss = self.loss_fn(y_hat, y) + self.L1_lambda * L1_reg

        return loss
