
# Import the required libraries
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# Define a new class called dp_reg that inherits from torch.nn.Module
class dp_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp", local_reg=False, threshold_based=True):
        super(dp_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        self.threshold_based = threshold_based

    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt, pct_a=0.0, pct_b=1.0):
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        # Sort probabilities in ascending order
        sorted_y0, _ = torch.sort(y0)
        sorted_y1, _ = torch.sort(y1)

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Local regularization
        if self.local_reg:
            if self.threshold_based:
                # Check if either of the filtered arrays is empty
                if len(sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]) == 0 or len(sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]) == 0:
                    raise ValueError(f"At least one group does not have predictions predictions in [{pct_a},  {pct_b}]. Impossible to regularize with [threshold_based==True]")

                reg_loss = torch.abs(torch.mean(sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]) - torch.mean(sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]))

            else:
                # Calculate the pct_a and pct_b percentile indices. Regularize on values inbetween.
                index_a_0 = int(pct_a * len_y0)
                index_b_0 = int(pct_b * len_y0)
                index_a_1 = int(pct_a * len_y1)
                index_b_1 = int(pct_b * len_y1)
                reg_loss = torch.abs(torch.mean(sorted_y0[index_a_0:index_b_0]) - torch.mean(sorted_y1[index_a_1:index_b_1]))

        # Global regularization [0,1]
        else:
            reg_loss = torch.abs(torch.mean(sorted_y0) - torch.mean(sorted_y1))

        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


class Wasserstein_reg(torch.nn.Module):
    """
    As implemented by Shalit et al., translated to PyTorch: https://github.com/clinicalml/cfrnet/blob/master/cfr/util.py
    """
    def __init__(self, mode="dp", local_reg=True, threshold_based=True, top_percentile=0.3):
        super(Wasserstein_reg, self).__init__()
        self.mode = mode
        self.local_reg = local_reg
        # Set whether regularization happens between two cut-off values (True) or two percentiles (False)
        self.threshold_based = threshold_based
        self.top_percentile = top_percentile

    def forward(self, y_pred, s, y_gt, pct_a, pct_b):

        top_percentile=self.top_percentile

        #Default settings of Shalit et al.:
        lam = 10
        its = 10
        sq = False
        backpropT = False

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]

        nc = float(y0.shape[0])
        nt = float(y1.shape[0])

        len_y0 = len(y0)
        len_y1 = len(y1)

        # Determine the length of the larger tensor
        max_len = max(len_y0, len_y1)

        y0_shape = y0.shape

        if self.local_reg:
            # Sort probabilities in ascending order
            sorted_y0, _ = torch.sort(y0)
            sorted_y1, _ = torch.sort(y1)

            if self.threshold_based:
                #only select values between \tau & 1
                y0 = sorted_y0[(sorted_y0 >= pct_a) & (sorted_y0 <= pct_b)]
                y1 = sorted_y1[(sorted_y1 >= pct_a) & (sorted_y1 <= pct_b)]
                if len(y0) == 0 or len(y1) == 0:
                    raise ValueError(f"At least one group does not have predictions in [{pct_a},  {pct_b}]. "
                                     f"Impossible to regularize with [threshold_based==True]")
            else:
                index_a_0 = int((1-top_percentile) * len_y0)
                index_b_0 = int(1 * len_y0)
                index_a_1 = int((1-top_percentile) * len_y1)
                index_b_1 = int(1 * len_y1)

                y0 = sorted_y0[index_a_0:index_b_0]
                y1 = sorted_y1[index_a_1:index_b_1]



        # Compute distance matrix
        M = torch.sqrt(torch.cdist(y1.unsqueeze(1), y0.unsqueeze(1), p=2) ** 2) #cdist requires at lest 2D tensor (and received 1D)

        # Estimate lambda and delta
        M_mean = torch.mean(M)

        M_drop = F.dropout(M, p=0.5)  # You can adjust the dropout rate if needed
        delta = torch.max(M).detach()  # Detach to prevent gradients from flowing
        eff_lam = (lam / M_mean).detach()  # Detach to prevent gradients from flowing

        # Compute new distance matrix with additional rows and columns
        row = delta * torch.ones((1, M.shape[1]), device=M.device)
        col = torch.cat((delta * torch.ones((M.shape[0], 1), device=M.device), torch.zeros((1, 1), device=M.device)),
                        dim=0)
        Mt = torch.cat((M, row), dim=0)
        Mt = torch.cat((Mt, col), dim=1)

        # Compute marginal vectors for treated and control groups
        p = 0.5 #In original code: given as parameter. Now just fixed on 0.5

        a_indices = torch.where(s > 0)[0]
        a = torch.cat([(p * torch.ones(len(y1)) / nt).unsqueeze(1), (1 - p) * torch.ones((1, 1))], dim=0)

        b_indices = torch.where(s < 1)[0]
        b = torch.cat([((1 - p) * torch.ones(len(y0)) / nc).unsqueeze(1), p * torch.ones((1, 1))], dim=0)

        # Compute kernel matrix and related matrices
        Mlam = eff_lam * Mt
        K = torch.exp(-Mlam) + 1e-6  # Added constant to avoid nan
        U = K * Mt
        ainvK = K / a

        # Compute u matrix iteratively
        u = a
        for i in range(its):
            u = 1.0 / torch.matmul(ainvK, (b / torch.t(torch.matmul(torch.t(u), K))))

        # Compute v matrix
        v = b / torch.t(torch.matmul(torch.t(u), K))

        # Compute transportation matrix T
        T = u * (torch.t(v) * K)

        if not backpropT:
            T = T.detach()  # Detach T if backpropagation is not needed

        # Compute E matrix and final Wasserstein distance D
        E = T * Mt
        D = 2 * torch.sum(E)

        #define reg_loss
        reg_loss = D

        torchtensor = torch.Tensor([0])

        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


