# based on https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/losses/vqperceptual.py
import torch
from torch import nn
import torch.nn.functional as F
from lpips import LPIPS


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (torch.mean(torch.nn.functional.softplus(-logits_real)) +
                    torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class KLLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, kl_weight=1.0, pixelloss_weight=1.0, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, disc_conditional=False, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is None:
            return torch.tensor(0.0)
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, discriminator, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, weights=None):
        res_ratio = 65536 / (inputs.shape[2] * inputs.shape[3])
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss + self.perceptual_weight * p_loss
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0] * res_ratio
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0] * res_ratio
        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss) * res_ratio

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"total_loss": loss.clone().detach().mean(),
                   "kl_loss": kl_loss.detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "p_loss": p_loss.detach().mean(),
                   "d_weight": d_weight.detach(),
                   "disc_factor": torch.tensor(disc_factor),
                   "g_loss": g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = discriminator(inputs.contiguous().detach())
                logits_fake = discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log