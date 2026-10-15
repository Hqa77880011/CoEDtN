import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from torch import nn
from torch.nn import functional as F


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


PORTS = np.array([0, 3, 12, 15])
INTERIOR = np.array([i for i in range(16) if i not in PORTS])
XY = np.array([(i / 3, j / 3) for i in range(4) for j in range(4)])
EDGES = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def reference(g, power):
    lap = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            a = i * 4 + j
            for di, dj, conductance in [
                (1, 0, g[0] * g[2] / g[1]),
                (0, 1, g[0] * g[1] / g[2]),
            ]:
                if i + di < 4 and j + dj < 4:
                    b = (i + di) * 4 + j + dj
                    lap[a, a] += conductance
                    lap[b, b] += conductance
                    lap[a, b] -= conductance
                    lap[b, a] -= conductance
    lii = lap[np.ix_(INTERIOR, INTERIOR)]
    lib = lap[np.ix_(INTERIOR, PORTS)]
    lift = np.zeros((16, 4))
    lift[PORTS] = np.eye(4)
    lift[INTERIOR] = -np.linalg.solve(lii, lib)
    source = np.zeros(16)
    source[INTERIOR] = power / len(INTERIOR)
    offset = np.zeros(16)
    offset[INTERIOR] = np.linalg.solve(lii, source[INTERIOR])
    k = lap[np.ix_(PORTS, PORTS)] + lib.T @ lift[INTERIOR]
    h = -lib.T @ offset[INTERIOR]
    return (k, h, lift, offset)


class PortModel(nn.Module):

    def __init__(self, direct=False, shared=True):
        super().__init__()
        self.direct = direct
        self.shared = shared
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(4, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                    nn.SiLU(),
                    nn.Linear(64, 20 if direct else 10),
                )
                for _ in range(1 if shared else 3)
            ]
        )
        self.decoder = nn.Sequential(nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 1))
        b = torch.zeros(6, 4)
        for row, (i, j) in enumerate(EDGES):
            b[row, i], b[row, j] = (1, -1)
        self.register_buffer("b", b)
        self.register_buffer("pi", torch.eye(4) - torch.ones(4, 4) / 4)

    def forward(self, g, power):
        features = torch.cat([g[:, :3].log(), power[:, None]], 1)
        raw = self.encoders[0](features)
        if not self.shared:
            family = g[:, 3].long().clamp(0, 2)
            raw = torch.stack([enc(features) for enc in self.encoders], 1)[
                torch.arange(len(g)), family
            ]
        if self.direct:
            return (raw[:, :16].reshape(-1, 4, 4), raw[:, 16:])
        w = F.softplus(raw[:, :6])
        k = torch.einsum("ei,ne,ej->nij", self.b, w, self.b) + 1e-05 * self.pi
        h = raw[:, 6:]
        h = h + (power - h.sum(1))[:, None] / 4
        return (k, h)

    def field(self, g, power, t, lift):
        xy = torch.tensor(XY, dtype=g.dtype, device=g.device).expand(len(g), -1, -1)
        desc = torch.cat(
            [g[:, :3].log(), power[:, None], t - t.mean(1, keepdim=True)], 1
        )
        inputs = torch.cat([xy, desc[:, None].expand(-1, 16, -1)], 2)
        distance = xy[..., 0] * (1 - xy[..., 0]) + xy[..., 1] * (1 - xy[..., 1])
        return torch.einsum("nij,nj->ni", lift, t) + distance * self.decoder(
            inputs
        ).squeeze(-1)


def prepare(args):
    rng = np.random.default_rng(args.seed)
    rows = []
    for parent in range(args.parents):
        family = parent % 3
        g = np.r_[np.exp(rng.uniform(-0.5, 0.5, 3)), family]
        power = rng.uniform(0.1, 2)
        k, h, lift, offset = reference(g, power)
        split = "component"
        if family != 2:
            split = rng.choice(["train", "validation", "test"], p=[0.7, 0.15, 0.15])
        for _ in range(args.excitations):
            t = rng.normal(300, 10, 4)
            rows.append(
                (g, power, t, -k @ t + h, lift @ t + offset, lift, split, parent)
            )
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data,
        **{
            key: np.asarray([r[i] for r in rows])
            for i, key in enumerate(
                ["g", "power", "t", "q", "field", "lift", "split", "parent"]
            )
        },
    )


def train(args, cfg):
    torch.manual_seed(args.seed)
    data = np.load(args.data)
    model = PortModel(args.variant == "direct", args.variant != "unshared")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    ids = np.flatnonzero(data["split"] == "train")
    val = np.flatnonzero(data["split"] == "validation")
    if not len(ids) or not len(val):
        raise ValueError(
            "Training and validation parent partitions must both be nonempty"
        )
    arrays = {
        k: torch.tensor(data[k], dtype=torch.float32)
        for k in ["g", "power", "t", "q", "field", "lift"]
    }
    best, wait = (float("inf"), 0)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in torch.randperm(len(ids)).split(cfg["batch_size"]):
            ix = ids[batch.numpy()]
            g, p, t, q, y, lift = [
                arrays[k][ix] for k in ["g", "power", "t", "q", "field", "lift"]
            ]
            k, h = model(g, p)
            pred = -torch.einsum("nij,nj->ni", k, t) + h
            loss = F.mse_loss(pred, q) + cfg["field_weight"] * F.mse_loss(
                model.field(g, p, t, lift), y
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            k, h = model(arrays["g"][val], arrays["power"][val])
            vl = F.mse_loss(
                -torch.einsum("nij,nj->ni", k, arrays["t"][val]) + h, arrays["q"][val]
            ).item()
        history.append({"epoch": epoch, "validation_port_mse": vl})
        if vl < best:
            best, wait = (vl, 0)
            torch.save(
                {"state": model.state_dict(), "variant": args.variant},
                Path(args.output) / "model.pt",
            )
        else:
            wait += 1
        if wait >= cfg["patience"]:
            break
    (Path(args.output) / "training.json").write_text(
        json.dumps(json_ready(history), indent=2, allow_nan=False)
    )


def assemble(ks, hs, ids, lifts, offsets, robin):
    n = int(np.max(ids)) + 1
    rr, cc, vv = ([], [], [])
    rhs = np.full(n, robin * 300.0)
    for k, h, ix in zip(ks, hs, ids):
        rr.extend(np.repeat(ix, 4))
        cc.extend(np.tile(ix, 4))
        vv.extend(k.ravel())
        np.add.at(rhs, ix, h)
    rr.extend(range(n))
    cc.extend(range(n))
    vv.extend(np.full(n, robin))
    matrix = coo_matrix((vv, (rr, cc)), shape=(n, n)).tocsr()
    temperatures = spsolve(matrix, rhs)
    port = temperatures[ids]
    fields = np.einsum("nij,nj->ni", lifts, port) + offsets
    flux = -np.einsum("nij,nj->ni", ks, port) + hs
    residual = np.zeros(n)
    for ix, q in zip(ids, flux):
        np.add.at(residual, ix, q)
    residual -= robin * (temperatures - 300)
    return (port, fields, flux, residual)


def evaluate(args, cfg):
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = PortModel(
        checkpoint["variant"] == "direct", checkpoint["variant"] != "unshared"
    )
    model.load_state_dict(checkpoint["state"])
    model.eval()
    rng = np.random.default_rng(args.seed)
    rows = []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for width in cfg["topology_widths"]:
        ids = np.array([[i, i + 1, width + 1 + i, width + 2 + i] for i in range(width)])
        g = np.column_stack(
            [
                np.exp(rng.uniform(-0.5, 0.5, (width, 3))),
                np.full(width, 2 if args.component_holdout else 0),
            ]
        )
        if args.component_holdout:
            g[:, 0] *= cfg["unseen_conductivity_factor"]
        power = rng.uniform(0.1, 2, width)
        ref = [reference(x, p) for x, p in zip(g, power)]
        rk, rh, lift, offset = [np.stack([r[j] for r in ref]) for j in range(4)]
        rt, truth, _, _ = assemble(rk, rh, ids, lift, offset, cfg["robin"])
        full_pred = None
        for variant in ["full", "no-solve", "identity-shuffle", "reference-DtN"]:
            start = time.perf_counter()
            with torch.no_grad():
                tg, tp = (
                    torch.tensor(g, dtype=torch.float32),
                    torch.tensor(power, dtype=torch.float32),
                )
                kg, hg = model(tg, tp)
                kg, hg = (kg.numpy(), hg.numpy())
                if variant == "reference-DtN":
                    kg, hg = (rk, rh)
                if variant == "identity-shuffle":
                    kg = kg[::-1].copy()
                pt, _, flux, residual = assemble(
                    kg, hg, ids, lift, np.zeros_like(offset), cfg["robin"]
                )
                if variant == "no-solve":
                    pt = np.full_like(pt, 300.0)
                    flux = -np.einsum("nij,nj->ni", kg, pt) + hg
                    residual = np.zeros(np.max(ids) + 1)
                    for ix, q in zip(ids, flux):
                        np.add.at(residual, ix, q)
                pred = (
                    truth
                    if variant == "reference-DtN"
                    else model.field(
                        tg,
                        tp,
                        torch.tensor(pt, dtype=torch.float32),
                        torch.tensor(lift, dtype=torch.float32),
                    ).numpy()
                )
            latency = (time.perf_counter() - start) * 1000
            if variant == "full":
                full_pred = pred.copy()
            xy = (
                np.concatenate([XY + [0, i] for i in range(width)])
                * cfg["component_size_mm"]
            )
            hotspot = np.linalg.norm(xy[np.argmax(pred)] - xy[np.argmax(truth)])
            component_gap = (
                100
                * np.linalg.norm(kg - rk, axis=(-2, -1))
                / np.maximum(np.linalg.norm(rk, axis=(-2, -1)), 1e-12)
            )
            rows.append(
                {
                    "width": width,
                    "variant": variant,
                    "nrmse_pct": float(
                        100
                        * np.sqrt(np.mean((pred - truth) ** 2))
                        / max(np.ptp(truth), 1e-12)
                    ),
                    "hotspot_mm": float(hotspot),
                    "gap_score": float(np.mean(component_gap)),
                    "component_gap_scores": component_gap.tolist(),
                    "flux_residual_pct": float(
                        100 * np.linalg.norm(residual) / max(np.sum(power), 1e-12)
                    ),
                    "energy_residual_pct": float(
                        100 * abs(flux.sum() - power.sum()) / power.sum()
                    ),
                    "latency_ms": latency,
                }
            )
        np.savez(
            output / f"fields_{width}.npz",
            prediction=full_pred,
            reference=truth,
            coordinates_mm=xy,
        )
    (output / "metrics.json").write_text(
        json.dumps(json_ready(rows), indent=2, allow_nan=False)
    )
    for variant in sorted({r["variant"] for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        plt.plot(
            [r["width"] for r in sub],
            [r["nrmse_pct"] for r in sub],
            marker="o",
            label=variant,
        )
    plt.xlabel("Component count")
    plt.ylabel("Field NRMSE (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "topology.png")
    plt.close()
    print(
        json.dumps(
            json_ready(
                {
                    "p95_latency_ms": float(
                        np.quantile(
                            [r["latency_ms"] for r in rows if r["variant"] == "full"],
                            0.95,
                        )
                    )
                }
            ),
            allow_nan=False,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/components.npz")
    parser.add_argument("--output", default="outputs/full")
    parser.add_argument("--model", default="outputs/full/model.pt")
    parser.add_argument(
        "--variant", choices=["energy", "direct", "unshared"], default="energy"
    )
    parser.add_argument("--parents", type=int, default=300)
    parser.add_argument("--excitations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--component-holdout", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args, cfg)
    else:
        evaluate(args, cfg)


if __name__ == "__main__":
    main()
