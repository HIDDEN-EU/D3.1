"""
Self-tests for every Levy hook. CPU only, no dataset, a couple of seconds.

    python -m opencood.models.LevyHooks.selftest

These assert the properties the dossier relies on, so that a claim made in prose
fails loudly if the code drifts: softmax parity, fp32 overflow safety at
gamma=4, the gate calibration table, that ray disagreement vanishes under
depth-independent context, and that cameras-only gating is an exact no-op
without cameras.
"""

import math

import torch
import torch.nn.functional as F

from opencood.models.LevyHooks import (GatewayHook, LevyDepthPool,
                                       LevyFeatureGateway,
                                       LevyIntraModalityFusion,
                                       LevyTemporalAggregator,
                                       build_depth_pool, build_gateway_hook,
                                       build_intra_modality_fusion,
                                       reset_registry, summary)


def test_depth_pool():
    logits = torch.randn(2, 41, 8, 22)

    # gamma=1 in logit mode must reproduce the LSS softmax exactly.
    pool = LevyDepthPool(score_mode='logit', gamma=1.0)
    w, ev, _, _ = pool(logits)
    assert torch.allclose(w, F.softmax(logits, dim=1), atol=1e-6), "softmax parity"
    assert torch.allclose(w.sum(1), torch.ones_like(w.sum(1)), atol=1e-5)

    # Evidence is tau * sum_k exp(S_k) -- unnormalised, so a flat ray and a
    # spiked ray with the same weights are told apart.
    flat = torch.zeros(1, 41, 1, 1)
    spike = torch.full((1, 41, 1, 1), -5.0)
    spike[0, 20] = 5.0
    _, ev_flat, _, _ = pool(flat)
    _, ev_spike, _, _ = pool(spike)
    assert ev_spike.item() > ev_flat.item(), "spiked ray must carry more mass"

    # LSS layout [B*N, C, D, fH, fW].
    ctx = torch.randn(2, 64, 41, 8, 22)
    pool_dis = LevyDepthPool(score_mode='logit', gamma=1.0,
                             return_disagreement=True)
    _, _, pooled, ray_dis = pool_dis(logits, ctx)
    assert pooled.shape == (2, 64, 8, 22), pooled.shape
    assert ray_dis.shape == (2, 1, 8, 22), ray_dis.shape
    assert (ray_dis >= 0).all() and (ray_dis <= 1).all()

    # HEAL's own LSS makes every bin along a ray a scalar multiple of one image
    # feature, so all bins share a direction and ray disagreement vanishes.
    shared = torch.randn(2, 64, 1, 8, 22) * torch.rand(2, 1, 41, 8, 22)
    _, _, _, dis_shared = pool_dis(logits, shared)
    assert dis_shared.abs().max().item() < 1e-4, \
        "depth-independent context must give zero ray disagreement"

    # gamma=4 on a large logit must not overflow.
    big = torch.tensor([[[[40.0]], [[-40.0]]]])
    w_big, _, _, _ = LevyDepthPool(score_mode='logit', gamma=4.0)(big)
    assert torch.isfinite(w_big).all(), "log-space weights must stay finite"

    # Builder contract: absent/disabled -> None.
    assert build_depth_pool(None) is None
    assert build_depth_pool({'enable': False}) is None
    assert build_depth_pool({'enable': True}) is not None
    print("  depth_pool          ok")


def test_feature_gateway():
    gw = LevyFeatureGateway(gate_power=0.5, score_mode='heal', gamma=4.0)
    occ = torch.tensor([[[[-20.0, 20.0]]]])          # sigmoid -> ~0, ~1
    feat = torch.ones(1, 8, 1, 2)
    gated, mask, lam = gw(feat, occ)
    assert abs(lam[0, 0, 0, 0].item() - 0.5) < 1e-3, "background Lambda ~ tau"
    assert abs(lam[0, 0, 0, 1].item() - 0.5 * math.e ** 4) < 1e-1, "fg Lambda"
    g_bg = gated[0, 0, 0, 0].item()
    g_fg = gated[0, 0, 0, 1].item()
    assert abs(g_bg - (1 - math.exp(-0.5)) ** 0.5) < 1e-5
    assert g_fg > 0.999 and g_bg < 0.63, f"gate contrast {g_bg:.3f} -> {g_fg:.3f}"

    # gate_ref renormalises so the reference score passes at unity.
    gw_ref = LevyFeatureGateway(gate_power=1.0, score_mode='logit', gamma=0.25,
                                gate_ref=0.0)
    gated_ref, _, _ = gw_ref(torch.ones(1, 4, 1, 1), torch.zeros(1, 1, 1, 1))
    assert abs(gated_ref.mean().item() - 1.0) < 1e-5, "gate_ref unity"
    print("  feature_gateway     ok")


def test_gateway_policy():
    """Who gets gated: levels, ego exemption, and the cameras-only no-op."""
    gate = LevyFeatureGateway(gate_power=0.5, score_mode='heal', gamma=4.0)
    feat = torch.randn(3, 8, 4, 4)
    occ = torch.randn(3, 1, 4, 4)
    record_len = [3]
    modalities = ['m1', 'm4', 'm3']          # ego lidar, camera, lidar
    cam_info = {'m4': {}}

    # levels: a level outside the set is returned untouched.
    hook = GatewayHook(gate, num_levels=3, levels=[0])
    hook.begin_forward()
    assert torch.equal(hook.apply(1, feat, occ, record_len), feat), \
        "ungated level must pass through unchanged"
    assert hook.stats == [], "an ungated level records no bandwidth stat"
    out0 = hook.apply(0, feat, occ, record_len)
    assert not torch.equal(out0, feat) and len(hook.stats) == 1

    # ego_exempt: agent 0 of each record_len group survives untouched.
    hook = GatewayHook(gate, num_levels=1, ego_exempt=True)
    hook.begin_forward()
    out = hook.apply(0, feat, occ, record_len)
    assert torch.equal(out[0], feat[0]), "ego must not be gated"
    assert not torch.equal(out[1], feat[1]), "collaborators must be gated"

    # cameras_only: only m4 is gated.
    hook = GatewayHook(gate, num_levels=1, cameras_only=True)
    hook.begin_forward()
    out = hook.apply(0, feat, occ, record_len, agent_modality_list=modalities,
                     cam_crop_info=cam_info)
    assert torch.equal(out[0], feat[0]), "lidar ego untouched"
    assert torch.equal(out[2], feat[2]), "lidar collaborator untouched"
    assert not torch.equal(out[1], feat[1]), "camera agent gated"

    # cameras_only with no cameras present is an EXACT no-op -- the property that
    # makes one configuration safe on every alliance composition.
    hook = GatewayHook(gate, num_levels=1, cameras_only=True)
    hook.begin_forward()
    out = hook.apply(0, feat, occ, record_len,
                     agent_modality_list=['m1', 'm3', 'm3'], cam_crop_info={})
    assert torch.equal(out, feat), "no cameras => bit-identical passthrough"

    # Builder contract.
    assert build_gateway_hook(None, num_levels=3) is None
    assert build_gateway_hook({'enable': False}, num_levels=3) is None
    assert build_gateway_hook({'enable': True, 'score_mode': None},
                              num_levels=3, default_score_mode='logit') is not None
    print("  gateway policy      ok")


def test_temporal():
    agg = LevyTemporalAggregator(in_channels=16, score_mode='heal', gamma=4.0)
    hist = torch.randn(2, 5, 16, 32, 32)
    fused, ev, dis = agg(hist)
    assert fused.shape == (2, 16, 32, 32) and ev.shape == (2, 1, 32, 32)
    assert dis.shape == (2, 1, 32, 32)
    assert (dis >= 0).all() and (dis <= 1).all(), "cosine disagreement bounds"

    same = torch.randn(1, 1, 16, 8, 8).repeat(1, 4, 1, 1, 1)
    _, _, dis_same = agg(same)
    assert dis_same.abs().max().item() < 1e-4, "repeated frames must not conflict"

    flip = torch.cat([same[:, :2], -same[:, :2]], dim=1)
    _, _, dis_flip = agg(flip)
    assert dis_flip.max().item() > 0.5, "antiparallel frames must conflict"
    print("  temporal            ok")


def test_intra_modality():
    imf = LevyIntraModalityFusion(in_channels=16, num_modalities=2,
                                  score_mode='heal', gamma=4.0)
    lidar, cam = torch.randn(3, 16, 24, 24), torch.randn(3, 16, 24, 24)
    fused, ev, dis = imf([lidar, cam])
    assert fused.shape == (3, 16, 24, 24)
    assert (dis >= 0).all() and (dis <= 1).all()

    # With explicit equal scores the fusion is the plain mean.
    eq = torch.zeros(3, 2, 1, 24, 24)
    fused_eq, _, _ = imf(torch.stack([lidar, cam], 1), scores=eq)
    assert torch.allclose(fused_eq, 0.5 * (lidar + cam), atol=1e-5), \
        "equal-score mean"

    args = {'in_channels': 128, 'out_channels': 64}
    assert build_intra_modality_fusion(None, args) is None
    assert build_intra_modality_fusion({'enable': True}, args) is not None
    try:
        build_intra_modality_fusion({'enable': True},
                                    {'in_channels': 100, 'out_channels': 64})
    except AssertionError:
        pass
    else:
        raise AssertionError("asymmetric channel counts must be rejected")
    print("  intra_modality      ok")


def main():
    torch.manual_seed(0)
    print("LevyHooks self-tests")
    test_depth_pool()
    test_feature_gateway()
    test_gateway_policy()
    test_temporal()
    test_intra_modality()
    reset_registry()
    print("all passed\n")
    print(summary())


if __name__ == '__main__':
    main()
