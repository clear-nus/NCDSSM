import torch
import torch.nn as nn

from ncdssm.models import NCDSSMLTI, NCDSSMLL, NCDSSMNL
from ncdssm.modules import MLP, ImageEncoder, ImageDecoder, MergeLastDims
from ncdssm.models.components import AuxInferenceModel, GaussianOutput, BernoulliOutput
from ..utils import get_activation


def build_model(config):
    if config["model"] == "NCDSSMLTI":
        aux_inf_base_net = ImageEncoder(
            img_size=config["img_size"],
            channels=1,
            out_dim=config["inference_img_enc_dim"],
        )
        inf_out_dim = (
            config["aux_dim"] if config["inference_tied_cov"] else 2 * config["aux_dim"]
        )
        aux_inf_dist_net = GaussianOutput(
            nn.Linear(aux_inf_base_net.out_dim, inf_out_dim),
            dist_dim=config["aux_dim"],
            use_tied_cov=config["inference_tied_cov"],
            use_trainable_cov=config["inference_trainable_cov"],
        )
        aux_inference_net = AuxInferenceModel(
            aux_inf_base_net,
            aux_inf_dist_net,
            aux_dim=config["aux_dim"],
            concat_mask=False,
        )

        y_emission_net = BernoulliOutput(
            nn.Sequential(
                ImageDecoder(
                    in_dim=config["aux_dim"],
                    img_size=config["img_size"],
                    channels=1,
                ),
                MergeLastDims(ndims=3),
            ),
            dist_dim=config["y_dim"],
            use_indepedent=False,
        )

        H = torch.eye(config["aux_dim"], config["z_dim"]) if config["fixed_H"] else None

        model = NCDSSMLTI(
            aux_inference_net,
            y_emission_net,
            aux_dim=config["aux_dim"],
            z_dim=config["z_dim"],
            y_dim=config["y_dim"],
            u_dim=config["u_dim"],
            integration_step_size=config["integration_step_size"],
            integration_method=config["integration_method"],
            H=H,
        )
    elif config["model"] == "NCDSSMLL":
        aux_inf_base_net = ImageEncoder(
            img_size=config["img_size"],
            channels=1,
            out_dim=config["inference_img_enc_dim"],
        )
        inf_out_dim = (
            config["aux_dim"] if config["inference_tied_cov"] else 2 * config["aux_dim"]
        )
        aux_inf_dist_net = GaussianOutput(
            nn.Linear(aux_inf_base_net.out_dim, inf_out_dim),
            dist_dim=config["aux_dim"],
            use_tied_cov=config["inference_tied_cov"],
            use_trainable_cov=config["inference_trainable_cov"],
        )
        aux_inference_net = AuxInferenceModel(
            aux_inf_base_net,
            aux_inf_dist_net,
            aux_dim=config["aux_dim"],
            concat_mask=False,
        )

        y_emission_net = BernoulliOutput(
            nn.Sequential(
                ImageDecoder(
                    in_dim=config["aux_dim"],
                    img_size=config["img_size"],
                    channels=1,
                ),
                MergeLastDims(ndims=3),
            ),
            dist_dim=config["y_dim"],
            use_indepedent=False,
        )

        alpha_net = nn.Sequential(
            MLP(
                in_dim=config["z_dim"],
                h_dim=config["alpha_mlp_units"],
                out_dim=config["K"],
                nonlinearity=get_activation(config["alpha_nonlinearity"]),
                n_hidden_layers=config["alpha_hidden_layers"],
            ),
            nn.Softmax(dim=-1),
        )

        H = torch.eye(config["aux_dim"], config["z_dim"]) if config["fixed_H"] else None

        model = NCDSSMLL(
            aux_inference_net,
            y_emission_net,
            K=config["K"],
            aux_dim=config["aux_dim"],
            z_dim=config["z_dim"],
            y_dim=config["y_dim"],
            u_dim=config["u_dim"],
            alpha_net=alpha_net,
            integration_step_size=config["integration_step_size"],
            integration_method=config["integration_method"],
            H=H,
        )
    elif config["model"] == "NCDSSMNL":
        aux_inf_base_net = ImageEncoder(
            img_size=config["img_size"],
            channels=1,
            out_dim=config["inference_img_enc_dim"],
        )
        inf_out_dim = (
            config["aux_dim"] if config["inference_tied_cov"] else 2 * config["aux_dim"]
        )
        aux_inf_dist_net = GaussianOutput(
            nn.Linear(aux_inf_base_net.out_dim, inf_out_dim),
            dist_dim=config["aux_dim"],
            use_tied_cov=config["inference_tied_cov"],
            use_trainable_cov=config["inference_trainable_cov"],
        )
        aux_inference_net = AuxInferenceModel(
            aux_inf_base_net,
            aux_inf_dist_net,
            aux_dim=config["aux_dim"],
            concat_mask=False,
        )

        y_emission_net = BernoulliOutput(
            nn.Sequential(
                ImageDecoder(
                    in_dim=config["aux_dim"],
                    img_size=config["img_size"],
                    channels=1,
                ),
                MergeLastDims(ndims=3),
            ),
            dist_dim=config["y_dim"],
            use_indepedent=False,
        )

        drift_net = MLP(
            in_dim=config["z_dim"],
            h_dim=config["drift_mlp_units"],
            out_dim=config["z_dim"],
            nonlinearity=get_activation(config["drift_nonlinearity"]),
            last_nonlinearity=config["drift_last_nonlinearity"],
            n_hidden_layers=config["drift_hidden_layers"],
            zero_init_last=config["drift_zero_init_last"],
            apply_spectral_norm=config["drift_spectral_norm"],
        )

        diffusion_nets = None
        if not config["fixed_diffusion"]:
            diffusion_nets = [
                nn.Sequential(
                    MLP(
                        in_dim=1,
                        h_dim=config["diffusion_mlp_units"],
                        out_dim=1,
                        nonlinearity=get_activation(config["diffusion_nonlinearity"]),
                        n_hidden_layers=config["diffusion_hidden_layers"],
                        apply_spectral_norm=config["diffusion_spectral_norm"],
                    ),
                    nn.Sigmoid(),
                )
                for _ in range(config["z_dim"])
            ]

        H = torch.eye(config["aux_dim"], config["z_dim"]) if config["fixed_H"] else None
        model = NCDSSMNL(
            aux_inference_net,
            y_emission_net,
            aux_dim=config["aux_dim"],
            z_dim=config["z_dim"],
            y_dim=config["y_dim"],
            u_dim=config["u_dim"],
            f=drift_net,
            gs=diffusion_nets,
            integration_step_size=config["integration_step_size"],
            integration_method=config["integration_method"],
            H=H,
        )
    else:
        raise ValueError(f'Unknown model {config["model"]}')
    return model
