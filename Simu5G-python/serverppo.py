import argparse, time, csv
import zmq
import numpy as np
import veinsgym_pb2 as pb

import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


class PolicyNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mu = nn.Linear(64, 1)
        self.v = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.5]))

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.v(h)



class PPOAgent:
    def __init__(self, obs_dim=6, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, device="cpu"):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.ent = ent
        self.device = device

        self.model = PolicyNet(obs_dim).to(device)
        self.opt = Adam(self.model.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []

    @torch.no_grad()
    def act(self, obs, greedy=False):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, v = self.model(x)
        std = torch.exp(self.model.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)

        if greedy:
            z = mu
        else:
            z = dist.rsample()

        u = torch.tanh(z)  # [-1,1]
        m = u + 1.0  # [0,2]

        # logprob with tanh correction
        logp_z = dist.log_prob(z).sum(dim=-1)
        logp = logp_z - torch.log(1 - u.pow(2) + 1e-6).sum(dim=-1)

        return float(m.item()), float(logp.item()), float(v.item()), float(u.item())

    def store(self, obs, act, logp, rew, val, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.done_buf.append(done)

    def _finish_path(self, last_val=0.0):
        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf + [last_val], dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        gae = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + self.gamma * vals[t+1] * nonterminal - vals[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae

        ret = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, epochs=10, batch_size=64):
        if len(self.obs_buf) < 32:
            self.reset_buffer()
            return

        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32, device=self.device)
        u = torch.tensor(np.array(self.act_buf), dtype=torch.float32, device=self.device).unsqueeze(1)  # [-1,1]
        old_logp = torch.tensor(np.array(self.logp_buf), dtype=torch.float32, device=self.device)

        adv, ret = self._finish_path(last_val=0.0)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.tensor(ret, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        idxs = np.arange(n)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, batch_size):
                mb = idxs[start:start + batch_size]

                mu, v = self.model(obs[mb])
                std = torch.exp(self.model.log_std).expand_as(mu)
                dist = torch.distributions.Normal(mu, std)

                # clamp u to avoid atanh blow-ups
                u_mb = torch.clamp(u[mb], -0.999999, 0.999999)

                # atanh(u) = 0.5 * (log(1+u) - log(1-u))
                z_mb = 0.5 * (torch.log1p(u_mb) - torch.log1p(-u_mb))

                logp_z = dist.log_prob(z_mb).sum(dim=-1)
                logp = logp_z - torch.log(1 - u_mb.pow(2) + 1e-6).sum(dim=-1)

                ratio = torch.exp(logp - old_logp[mb])

                clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                pi_loss = -(torch.min(ratio * adv[mb], clip_adv)).mean()

                v_loss = 0.5 * (ret[mb] - v.squeeze(1)).pow(2).mean()

                # entropy of the underlying Normal (good enough as an exploration bonus)
                ent = dist.entropy().sum(dim=-1).mean()

                loss = pi_loss + v_loss - self.ent * ent

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        self.reset_buffer()


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="ppo_policy.pt")
    ap.add_argument("--log", default="rollout_ppo.csv")
    ap.add_argument("--update_every", type=int, default=512)
    ap.add_argument("--save_every_episodes", type=int, default=1)
    ap.add_argument("--greedy_eval", action="store_true")
    args = ap.parse_args()

    set_seeds(args.seed)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[PPO] Listening on {args.host}:{args.port} | mode={args.mode} | seed={args.seed}")

    agent = PPOAgent(obs_dim=6)

    if args.mode == "eval":
        agent.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        agent.model.eval()
        print(f"[PPO] Loaded checkpoint {args.checkpoint}")

    logf = open(args.log, "a", newline="")
    w = csv.writer(logf)
    if logf.tell() == 0:
        w.writerow(["ts","episode","step","thr","delay","jitter","loss","numUe","stepEnergyJ","rew","act"])

    last_obs = None
    last_u = None
    last_logp = None
    last_val = None

    episode = 0
    steps = 0
    ep_return = 0.0

    while True:
        msg = sock.recv()
        req = pb.Request()
        req.ParseFromString(msg)

        reply = pb.Reply()
        reply.id = req.id

        if req.HasField("init"):
            print("[PPO] Init received")
            last_obs = None
            last_u = None
            last_logp = None
            last_val = None
            steps = 0
            ep_return = 0.0
            sock.send(reply.SerializeToString())
            continue

        if req.HasField("shutdown"):
            # mark terminal for last transition (if any)
            if args.mode == "train" and len(agent.done_buf) > 0:
                agent.done_buf[-1] = 1.0

            if args.mode == "train":
                agent.update()
                if (episode % args.save_every_episodes) == 0:
                    torch.save(agent.model.state_dict(), args.checkpoint)

            print(f"[PPO] Shutdown (episode {episode}) return={ep_return:.3f} steps={steps}")

            # increment episode counter
            episode += 1

            sock.send(reply.SerializeToString())
            continue

        if req.HasField("step"):
            obs = np.array(req.step.observation.box.values, dtype=np.float32)
            rew = 0.0
            if req.step.reward and req.step.reward.HasField("box") and len(req.step.reward.box.values) > 0:
                rew = float(req.step.reward.box.values[0])

            # store transition from previous action
            if last_obs is not None:
                if args.mode == "train":
                    agent.store(last_obs, last_u, last_logp, rew, last_val, done=0.0)
                ep_return += rew
                steps += 1

                if args.mode == "train" and len(agent.obs_buf) >= args.update_every:
                    agent.update()
                    print(f"[PPO] Update done @ episode={episode} step={steps} return≈{ep_return:.3f}")

            # pick next action
            greedy = (args.mode == "eval" and args.greedy_eval)
            m, logp, v, u = agent.act(obs, greedy=greedy)

            thr, delay, jitter, loss, numUe, stepE = obs.tolist()
            w.writerow([time.time(), episode, steps, thr, delay, jitter, loss, numUe, stepE, rew, m])
            logf.flush()

            last_obs = obs
            last_u = u
            last_logp = logp
            last_val = v

            if steps % 20 == 0:
                print(f"[PPO] ep={episode} step={steps} rew={rew:.4f} act(m)={m:.3f} obs0..2={obs[:3]}")

            reply.action.box.values[:] = [m]
            sock.send(reply.SerializeToString())
            continue

        sock.send(reply.SerializeToString())


if __name__ == "__main__":
    main()
