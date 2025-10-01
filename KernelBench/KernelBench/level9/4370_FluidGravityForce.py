import torch
import torch.nn as nn


class FluidGravityForce(nn.Module):

    def __init__(self, gravity, maxSpeed=3):
        """
        Initializes a fluid gravity model.

        Arguments:
            gravity: Gravity vector in the global frame (same as particle l) for the simulation
            maxSpeed: The maximum magnitude of the particle velocities. Higher velocities are clamped.
                      Previous work used: MAX_VEL = 0.5*0.1*NSUBSTEPS/DT
        """
        super(FluidGravityForce, self).__init__()
        self.gravity = gravity
        self.maxSpeed = maxSpeed
        self.relu = nn.ReLU()

    def _cap_magnitude(self, A, cap):
        d = len(A.size())
        vv = torch.norm(A, 2, d - 1, keepdim=True)
        vv = cap / (vv + 0.0001)
        vv = -(self.relu(-vv + 1.0) - 1.0)
        return A * vv

    def forward(self, locs, vel, dt):
        """
        Applies gravity force to fluid sim
        Inputs:
            locs: A BxNx3 tensor where B is the batch size, N is the number of particles.
                  The tensor contains the locations of every particle.
            vels: A BxNx3 tensor that contains the velocity of every particle
            dt: timestep to predict for
            gravity: 1x1x3 tensor containing the direction of gravity in the same coordinate frame as particles
            maxSpeed: maximum velocity possible for nay particle
        Returns:
            locs: A BxNx3 tensor with the new particle positions
            vel:  A BxNx3 tensor with the new particle velocities
        """
        vel = vel + self.gravity * dt
        vel = self._cap_magnitude(vel, self.maxSpeed)
        locs = locs + vel * dt
        return locs, vel


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'gravity': 4}]
