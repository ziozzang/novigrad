"""Experimental ordered sparse Metal kernels shared by the probe and wrapper.

Stable argsort: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.argsort.html
Custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
Dense mode is diagnostic only: floating reduction order fails Engine parity.
"""
import time
import mlx.core as mx
import numpy as np
from safetensors import safe_open


def sparse_operator(pre, post, weights, outputs):
    order = np.argsort(post, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(post, minlength=outputs))]).astype(np.uint32)
    indices = mx.array(np.asarray(pre[order], np.uint32))
    values = mx.array(np.asarray(weights[order], np.float32))
    offsets = mx.array(offsets)
    mx.eval(indices, values, offsets)
    kernel = mx.fast.metal_kernel(name="ordered_sparse", input_names=["x", "indices", "weights", "offsets"],
        output_names=["out"], header="#pragma clang fp contract(off)\n", source="""
        uint o = thread_position_in_grid.x;
        if (o >= x_shape[0] * N) return;
        uint row = o / N, post = o % N;
        float sum = 0.0f;
        for (uint e = offsets[post]; e < offsets[post+1]; ++e) {
            float product = weights[e] * x[row * x_shape[1] + indices[e]];
            sum = sum + product;
        }
        out[o] = sum;
        """, compile_options={"math_mode": "safe"})
    def apply(x):
        return kernel(inputs=[x, indices, values, offsets], template=[("N", outputs)],
            grid=(x.shape[0] * outputs, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(x.shape[0], outputs)], output_dtypes=[mx.float32])[0]
    return apply


def build(path, selection="argsort", include_dense=True):
    """Build kernels from a checkpoint already validated by the Rust Engine."""
    if selection not in ("argsort", "partition"):
        raise ValueError("selection must be argsort or partition")
    start = time.perf_counter()
    with safe_open(str(path), framework='numpy') as f:
        meta = f.metadata()
        t = {k: f.get_tensor(k) for k in f.keys()}
    if meta['format'] != 'nobi.plastic' or meta['version'] not in ('3', '4'):
        raise ValueError('expected Engine checkpoint version 3 or 4')
    ni, nh, no = (len(t[k]) for k in ('input_ids', 'hidden_ids', 'output_ids'))
    if include_dense and (ni * nh + nh * no) * 4 > 128 * 1024**2:
        raise ValueError('probe refuses dense matrices larger than 128 MiB')
    actions = int(meta['actions'])
    pre, post = t['input_pre'].astype(int), t['input_post'].astype(int)
    counts = t['input_count'].astype(np.float64)
    totals = np.bincount(post, weights=counts, minlength=nh)
    fixed = (counts / totals[post]).astype(np.float32) * t['input_sign']
    groups = t['output_actions'].astype(int)
    dense_bytes = 0
    if include_dense:
        first = np.zeros((ni, nh), np.float32)
        np.add.at(first, (pre, post), fixed)
        second = np.zeros((nh, no), np.float32)
        np.add.at(second, (t['plastic_pre'].astype(int), t['plastic_post'].astype(int)), t['plastic_weight'])
        decode = np.zeros((no, actions), np.float32)
        decode[np.arange(no), groups] = t.get('output_gains', np.ones(no, np.float32))
        a, b, c = map(mx.array, (first, second, decode))
        mx.eval(a, b, c)
        dense_bytes = first.nbytes + second.nbytes + decode.nbytes
    scales = np.float32(meta['logit_gain']) / np.bincount(groups, minlength=actions).astype(np.float32)
    keep = max(1, min(nh, int(np.ceil(np.float32(nh) * np.float32(meta['active_fraction'])))))
    scale = mx.array(scales)
    mx.eval(scale)
    setup = time.perf_counter() - start

    sparse_start = time.perf_counter()
    sparse_first = sparse_operator(pre, post, fixed, nh)
    sparse_second = sparse_operator(t["plastic_pre"].astype(int), t["plastic_post"].astype(int), t["plastic_weight"], no)
    sparse_decode = sparse_operator(np.arange(no), groups, t.get("output_gains", np.ones(no, np.float32)), actions)

    def forward(x, sparse=False):
        if not sparse and not include_dense:
            raise ValueError("dense matrices were not requested")
        magnitude = mx.max(mx.abs(x), axis=1, keepdims=True)
        x = x / mx.where(magnitude > 16, magnitude, mx.ones_like(magnitude))
        h = mx.maximum(sparse_first(x) if sparse else x @ a, 0)
        if selection == "partition":
            threshold = mx.partition(h, nh - keep, axis=1)[:, nh - keep:nh - keep + 1]
            above = h > threshold
            tied = h == threshold
            remaining = keep - mx.sum(above, axis=1, keepdims=True)
            mask = above | (tied & (mx.cumsum(tied.astype(mx.int32), axis=1) <= remaining))
        else:
            selected = mx.argsort(-h, axis=1)[:, :keep]
            mask = mx.put_along_axis(mx.zeros_like(h), selected, mx.ones(selected.shape), axis=1)
        h = h * mask
        maximum = mx.max(h, axis=1, keepdims=True)
        h = h / mx.where(maximum > 0, maximum, mx.ones_like(maximum))
        logits = (sparse_decode(sparse_second(h)) if sparse else (h @ b) @ c) * scale
        return mx.softmax(logits, axis=1)

    return forward, t['input_ids'], {'matrix_conversion_and_upload_seconds': setup,
        'sparse_conversion_and_upload_seconds': time.perf_counter() - sparse_start,
        'dense_matrix_bytes': dense_bytes,
        'shape': [ni, nh, no], 'keep': keep}


