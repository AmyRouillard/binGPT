import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.autograd import grad


class SaddleFreeNewton(optim.Optimizer):
    """ """

    def __init__(self, params, lr=1.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
        )
        super(SaddleFreeNewton, self).__init__(params, defaults)

    def step(self, closure):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. This is essential because the optimizer
                needs to compute gradients and Hessian-vector products.
        """
        # We require a closure to recompute gradients and for HVPs
        # Ensure the graph is created for the HVP calculation
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params = []
        for group in self.param_groups:
            params.extend(group["params"])

        if len(params) == 0:
            return loss

        # Flatten parameters
        params = [p for p in params if p.requires_grad]
        # flat_params = torch.cat([p.contiguous().view(-1) for p in params])

        grad = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad])

        # Hessian matrix
        hessian = []
        for g in flat_grad:
            second_grads = torch.autograd.grad(g, params, retain_graph=True)
            h_row = torch.cat([sg.contiguous().view(-1) for sg in second_grads])
            hessian.append(h_row)
        hessian = torch.stack(hessian)

        eigenvalues, eigenvectors = torch.linalg.eig(hessian)

        # construct the inverse Hessian from the eigenvalues
        inv_hessian_abs = torch.zeros_like(hessian)
        for i in range(len(eigenvalues)):
            tmp = (
                torch.outer(eigenvectors[:, i], eigenvectors[:, i])
                / eigenvalues[i].real.abs()
            )
            inv_hessian_abs += tmp.real

        flat_update = -(flat_grad @ inv_hessian_abs)
        # flat_update = -( inv_hessian_abs @ flat_grad )

        # clamp the update to avoid exploding gradients
        flat_update = torch.clamp(flat_update, -1e2, 1e2)

        # Reshape the update to match the parameter shapes
        update = []
        start = 0
        for p in params:
            end = start + p.numel()
            update.append(flat_update[start:end].view_as(p))
            start = end

        # update parameters using the inverse Hessian
        with torch.no_grad():
            for p, u in zip(params, update):
                p.add_(u, alpha=self.defaults["lr"])

        return loss, update
