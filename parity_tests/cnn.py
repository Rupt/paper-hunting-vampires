import torch
import torch.nn.functional as F


def get_loss_fn(loss_name="loss_ls"):
    """we have choice of various losses to use. We get as input the
    net outputs per batch, and return the loss (to be minimised)

    Note - we modify the final number in the net by an activation here"""
    if loss_name == "loss_ls":

        def loss_fn(outputs):
            loss = -torch.mean(
                torch.nn.LogSigmoid()(outputs)
            )  # equivalent negative log likelihood ratio which is translated by -np.log(0.5)
            return loss

    # other losses used for code development
    if loss_name == "loss_ls_meanstd":

        def loss_fn(outputs):
            stdDev_W, mean_W = torch.std_mean(torch.nn.LogSigmoid()(outputs))
            loss = -mean_W / (stdDev_W + 10e-6)
            return loss

    if loss_name == "loss_meanstd":

        def loss_fn(outputs):
            stdDev_W, mean_W = torch.std_mean(outputs)
            loss = -mean_W / (stdDev_W + 10e-6)
            return loss

    if loss_name == "loss_ss":

        def loss_fn(outputs):
            outputs = torch.nn.Softsign()(outputs)
            loss = -torch.mean(outputs)
            return loss

    if loss_name == "loss_sum":

        def loss_fn(outputs):
            loss = -torch.mean(outputs)
            return loss

    return loss_fn


class parity_odd_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 5
        self.kernel = (self.kernel_size, self.kernel_size)

        self.conv1 = torch.nn.Conv2d(1, 6, self.kernel_size)
        self.conv2 = torch.nn.Conv2d(6, 6, self.kernel_size)
        self.fc1 = torch.nn.Linear(96, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def phi_invariant_pool(self, x, eta_window):
        """pool over entire phi so be invariant
        x.shape[-1] == the phi axis shape"""
        return torch.nn.MaxPool2d((eta_window, x.shape[-1]))(x)

    def forward(self, x):
        """build in our into the overall network the symmetries"""

        # original event + exchaned two beams
        real = self.subnet(x) + self.subnet(rotate_180(x))

        # parity flipped on reak and exchanged two beams
        fake = self.subnet(parity_flip(x)) + self.subnet(
            parity_fliprotate_180(x)
        )

        # return the output without modifications for applying of different losses
        return real - fake

    def subnet(self, x):
        """forward pass"""

        #         print("1. Initial dims", x.size())
        x = pad_phi_eta(x, self.kernel_size)  # phi invariant pad
        #         print("2. ", x.size())
        x = self.conv1(x)
        x = torch.nn.LeakyReLU()(x)

        x = pad_phi_eta(x, self.kernel_size)
        x = self.conv2(x)
        #         print("3. ", x.size())
        #         x = torch.nn.ReLU()(x)
        x = torch.nn.LeakyReLU()(x)
        #         print("4. ", x.size())
        x = self.phi_invariant_pool(x, 2)
        #         print("5. ", x.size())
        x = torch.flatten(x, start_dim=1)
        #         print("6. ", x.size())
        x = self.fc1(x)
        x = torch.nn.LeakyReLU()(x)
        x = self.fc2(x)
        #         print("7. ", x.size())

        return x


def assert_n_eta_phi_c(arr):
    """needs to have 4 channels"""
    assert len(arr.shape) == 4


def flip_eta(x):
    assert_n_eta_phi_c(x)
    return torch.flip(x, [2])


def flip_phi(x):
    assert_n_eta_phi_c(x)
    return torch.flip(x, [3])


def rotate_180(x):
    """a rotation of 180degrees around the x axis, flipping our beams.
    This can be decomposed into a flip in eta and a flip in phi"""
    assert_n_eta_phi_c(x)
    return flip_eta(flip_phi(x))


def parity_fliprotate_180(x):
    """parity flipping our rotated by 180 beam is equivalent
    to a flip in phi"""
    assert_n_eta_phi_c(x)
    return flip_phi(x)


def parity_flip(x):
    """Note parity flip px,py,pz -> -px,-py,-pz
    gives eta -> - eta and phi -> phi +- pi, depending on the sign of phi

    Since our network is translationally invariant in phi, the parity operator
    has no effect on phi. So a parity flip is just a flip in eta"""
    assert_n_eta_phi_c(x)
    return flip_eta(x)


def roll_down_phi(x, distance):
    """roll down in phi in order to check the translation invariance
    of our network"""
    assert_n_eta_phi_c(x)
    return torch.roll(x, distance, 3)


def pad_phi(x, n, mode="circular"):
    """pad the top in phi for convolutions"""
    return F.pad(
        input=x,
        pad=(n, 0, 0, 0),
        mode=mode,  #'constant', 'reflect', 'replicate' or 'circular'.
    )


def pad_phi_TB(x, n, mode="circular"):
    """pad the top in phi for convolutions"""
    return F.pad(
        input=x,
        pad=(n, n, 0, 0),
        mode=mode,  #'constant', 'reflect', 'replicate' or 'circular'.
    )


def pad_phi_eta(x, kernel_size):
    """pad the top in phi for convolutions"""
    n = (
        kernel_size - 1
    ) // 2  # for keeping same size (works for odd kernel_size)
    x = pad_phi_TB(x, n, "circular")  # cyclical
    x = F.pad(
        input=x, pad=(0, 0, n, n), mode="constant", value=0.0
    )  # zero pad eta
    return x
