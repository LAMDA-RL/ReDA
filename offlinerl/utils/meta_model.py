import numpy as np
import torch
import torch.optim as optim
import os

from offlinerl.utils.function import product_of_gaussians

class MetaDynamicsAgent(object):
    def __init__(self, args, encoder, decoder, device):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.beta = args["beta"]
    
        self.optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=args['lr'], weight_decay=args['weight_decay'])
    
    def sample_product_z(self, context):
        encoder = self.encoder
        
        context = torch.from_numpy(context).to(self.device)
        z_means, z_vars, _ = encoder(context)
        
        # num_task, batch_size, latent_dim = z_mean.shape
        # z_params = [product_of_gaussians(mu, logvar) for mu, logvar in zip(torch.unbind(z_mean), torch.unbind(z_logvar))]
        # product_z_means = torch.stack([p[0] for p in z_params])
        # product_z_vars = torch.stack([p[1] for p in z_params])
        
        return z_means, z_vars
    
    def infer_posteriors(self, z_means, z_vars):
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
        z = torch.stack([d.rsample() for d in posteriors])
        return z

    def clear_grad(self):
        self.encoder_loss = 0.0
        self.model_loss = 0.0
        self.clip_loss = 0.0
        self.decoder_loss = 0.0

        self.model_mse = 0.0
        self.logstd = 0.0

        self.z_means_summary = 0.0
        self.z_vars_summary = 0.0

        self.encoder_cnt = 0
        self.decoder_cnt = 0

        self.optimizer.zero_grad()
        
    def cumulate_encoder_loss(self, z_means, z_logvars, z_vars):
        self.encoder_loss += -0.5 * torch.sum(1 + z_logvars - z_means**2 -  z_vars)
        self.encoder_cnt += 1

        self.z_means_summary += torch.mean(torch.abs(z_means))
        self.z_vars_summary += torch.mean(torch.abs(z_vars))

    def cumulate_decoder_loss(self, obs, action, z, next_obs, reward):
        obs = torch.from_numpy(obs).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        
        dist = self.decoder(torch.cat([obs, action], axis=-1), z)
        self.model_loss += (- dist.log_prob(torch.cat([next_obs, reward], axis=-1))).sum(dim=1).mean()
        if self.args["train_with_clip_loss"]:
            self.clip_loss += 0.01 * (2.0 * self.decoder.max_logstd).sum() - 0.01 * (2.0 * self.decoder.min_logstd).sum()
        else:
            self.clip_loss += 0.0
        
        assert dist.mean.shape == torch.cat([next_obs, reward], axis=-1).shape
        self.model_mse += ((dist.mean - torch.cat([next_obs, reward], axis=-1))**2).sum(dim=1).mean()
        self.logstd += dist.scale.cpu().mean()
        
        self.decoder_loss = self.model_loss + self.clip_loss
        self.decoder_cnt += 1
        
    def train(self):
        if self.encoder_cnt > 0:
            self.encoder_loss /= self.encoder_cnt
            self.z_means_summary /= self.encoder_cnt
            self.z_vars_summary /= self.encoder_cnt
        
        self.decoder_loss /= self.decoder_cnt
        self.model_mse /= self.decoder_cnt
        self.logstd /= self.decoder_cnt
        
        loss = self.beta*self.encoder_loss + self.decoder_loss
        loss.backward()
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(self.device) for p in self.encoder.parameters()]))
        self.optimizer.step()
        
        result = dict()
        if not isinstance(self.encoder_loss, float):
            result["encoder_loss"] = self.encoder_loss.detach().cpu().item()
        result["model_loss"] = (self.model_loss/self.decoder_cnt).detach().cpu().item()
        if not isinstance(self.clip_loss, float):
            result["clip_loss"] = (self.clip_loss/self.decoder_cnt).detach().cpu().item()
        result["decoder_loss"] = self.decoder_loss.detach().cpu().item()
        if not isinstance(self.z_means_summary, float):
            result["z_means_summary"] = self.z_means_summary.detach().cpu().item()
        if not isinstance(self.z_vars_summary, float):
            result["z_vars_summary"] = self.z_vars_summary.detach().cpu().item()
        result["mse_loss"] = self.model_mse.detach().cpu().item()
        result["mean_logstd"] = self.logstd.detach().cpu().item()
        # result["total_norm"] = total_norm.detach().cpu().item()
        return result
    
    def eval(self, obs, action, z, next_obs, reward):
        obs = torch.from_numpy(obs).to(self.device)
        action = torch.from_numpy(action).to(self.device)
        next_obs = torch.from_numpy(next_obs).to(self.device)
        reward = torch.from_numpy(reward).to(self.device)
        
        target = torch.cat([next_obs, reward], axis=-1)
        
        result = dict()
        
        dist = self.decoder(torch.cat([obs, action], axis=-1), z)
        model_loss = (- dist.log_prob(target)).sum(dim=1).mean()
        if self.args["train_with_clip_loss"]:
            clip_loss = 0.01 * (2.0 * self.decoder.max_logstd).sum() - 0.01 * (2.0 * self.decoder.min_logstd).sum()
        else:
            clip_loss = 0.0
        mse_loss = ((target - dist.mean)**2).sum(dim=1).mean()
        result['model_loss_eval'] = model_loss.detach().cpu().item()
        if isinstance(clip_loss, float):
            result['clip_loss_eval'] = clip_loss
        else:
            result['clip_loss_eval'] = clip_loss.detach().cpu().item()
        result['mean_logstd_eval'] = dist.scale.cpu().mean().item()
        result['mse_loss_eval'] = mse_loss.detach().cpu().item()

        return result
    
    def save(self, save_path):
        assert save_path, "save path cannot be None"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.encoder.state_dict(), os.path.join(save_path, "meta_encoder.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(save_path, "meta_decoder.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))

    def load(self, load_path):
        assert load_path, "load path cannot be None"
        # self.encoder.load_state_dict(torch.load(os.path.join(load_path, "meta_encoder.pt"), map_location=self.device))
        self.decoder.load_state_dict(torch.load(os.path.join(load_path, "meta_decoder.pt"), map_location=self.device))
        # self.optimizer.load_state_dict(torch.load(os.path.join(load_path, "optimizer.pt"), map_location=self.device))
