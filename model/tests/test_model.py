import torch

from model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32)


def test_output_shape_and_loss():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (3, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 16, CFG.vocab_size)
    # An untrained model should be about as surprised as a uniform guess.
    assert abs(loss.item() - torch.log(torch.tensor(float(CFG.vocab_size))).item()) < 0.5


def test_default_parameter_count():
    assert GPT(GPTConfig()).num_params() == 1_325_824


def test_causal_mask_blocks_the_future():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    a = torch.randint(0, CFG.vocab_size, (1, 16))
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % CFG.vocab_size
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:], atol=1e-5)


def test_trace_matches_plain_forward():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 12))
    trace = {}
    traced, _ = model(idx, trace=trace)
    plain, _ = model(idx)
    assert torch.allclose(traced, plain)
    assert len(trace["layers"]) == CFG.n_layer
    attn = trace["layers"][0]["attn"]
    assert attn.shape == (1, CFG.n_head, 12, 12)
    assert torch.allclose(attn.sum(-1), torch.ones(1, CFG.n_head, 12), atol=1e-5)


def test_ablations_change_the_output():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 12))
    base, _ = model(idx)
    assert not torch.allclose(base, model(idx, no_pos=True)[0], atol=1e-5)
    assert not torch.allclose(base, model(idx, ablate_heads=[(0, 1)])[0], atol=1e-5)
    assert not torch.allclose(base, model(idx, ablate_mlp=[1])[0], atol=1e-5)


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
