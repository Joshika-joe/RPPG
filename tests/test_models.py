import torch

from src.models import MODEL_NAMES, build_model, count_params, run_model


def test_registry_names():
    assert MODEL_NAMES == ["v2", "v3"]


def test_v3_output_shape_and_variable_length():
    m = build_model("v3").eval()
    for t in (60, 150):
        x = torch.randn(2, 3, t, 64, 64)
        a = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            y = m(x, appearance=a)
        assert y.shape == (2, t)
    # attention mask averages to 1 over cells, shape (B,1,8,8)
    assert m.last_attention.shape == (2, 1, 8, 8)
    assert torch.allclose(m.last_attention.mean(dim=(2, 3)), torch.ones(2, 1), atol=1e-4)


def test_v3_without_appearance_uses_uniform_mask():
    m = build_model("v3").eval()
    with torch.no_grad():
        m(torch.randn(1, 3, 30, 64, 64))
    assert torch.allclose(m.last_attention, torch.ones_like(m.last_attention))


def test_v2_output_shape():
    m = build_model("v2").eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 150, 64, 64))
    assert y.shape == (1, 150)


def test_param_counts():
    assert count_params(build_model("v3")) < count_params(build_model("v2")) / 2


def test_run_model_routes_mean_frame():
    m = build_model("v3").eval()
    x = torch.randn(2, 3, 30, 64, 64)
    meta = {"mean_frame": torch.rand(2, 3, 64, 64)}
    with torch.no_grad():
        y = run_model(m, x, meta, torch.device("cpu"))
    assert y.shape == (2, 30)
    assert not torch.allclose(m.last_attention, torch.ones_like(m.last_attention))


def test_v3_backward():
    m = build_model("v3").train()
    x = torch.randn(2, 3, 30, 64, 64)
    a = torch.rand(2, 3, 64, 64)
    loss = m(x, appearance=a).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
