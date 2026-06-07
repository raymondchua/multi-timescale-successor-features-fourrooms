import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict, freeze, unfreeze
from flax.linen.initializers import lecun_normal

def lecun_reinit(key, shape):
    return lecun_normal()(key, shape, dtype=jnp.float32)

Array = jax.Array

@struct.dataclass
class CBPLayerState:
    u: Array
    f: Array
    a: Array

@struct.dataclass
class CBPState:
    layers: tuple[CBPLayerState, ...]  # (rep_layer, hidden_layer)

def init_cbp_state(feature_dim: int, hidden_dim: int) -> CBPState:
    def layer(w):
        return CBPLayerState(
            u=jnp.zeros((w,), jnp.float32),
            f=jnp.zeros((w,), jnp.float32),
            a=jnp.zeros((w,), jnp.float32),
        )
    return CBPState(layers=(layer(feature_dim), layer(hidden_dim)))

def cbp_generate_and_test_dqn_head(
    key: Array,
    params: FrozenDict,
    cbp: CBPState,
    hs: list[Array],              # [rep, hid] each [B, width]
    *,
    dead_eps: float = 1e-3,
    rho: float = 1e-4,
    eta: float = 0.99,
    maturity: int = 100,
):
    """
    CBP on the MLP head only:
      l=0 => rep (outgoing to hidden)
      l=1 => hidden (outgoing to q)
    Assumes params contain "rep", "hidden", "q" Dense subtrees.
    """
    p = params
    k = key

    # replaced_rep = jnp.array(0, dtype=jnp.int32)
    # replaced_hid = jnp.array(0, dtype=jnp.int32)

    def process_layer(
        k_in: Array,
        st: "CBPLayerState",
        h: Array,
        W_in: Array,
        W_out: Array,
    ):
        """
        Args:
            st: CBPLayerState for this layer, fields [n_out]
            h: activations feeding this layer, [B, n_out]
            W_in: incoming weights into this layer's units (columns correspond to units)
            W_out: outgoing weights from this layer's units (rows correspond to units)

        Returns:
            k_out, W_in_new, W_out_new, st_new, replace_mask, eligible, k_replace_float, k_replace_int, replaced_count
        """
        a = st.a + 1.0  # [n_out]

        # Batch mean activation for units
        h_mean = jnp.mean(h, axis=0)  # [n_out]

        # EMA baseline
        f = (1.0 - eta) * h_mean + eta * st.f
        # Bias correction (safe since eta^a is well-defined for a>=1)
        f_hat = f / (1.0 - jnp.power(eta, a))

        # Surprise / utility proxy
        out_mag = jnp.sum(jnp.abs(W_out), axis=1)  # [n_out]
        in_mag = jnp.sum(jnp.abs(W_in), axis=0) + 1e-8  # [n_out]
        y = jnp.abs(h_mean - f_hat) * (out_mag / in_mag)  # [n_out]

        u = (1.0 - eta) * y + eta * st.u
        u_hat = u / (1.0 - jnp.power(eta, a))

        n_out = W_in.shape[1]
        eligible = a > float(maturity)  # [n_out] bool

        # Dead (uses your dead_eps hyperparam)
        dead_mask = jnp.abs(f_hat) < float(dead_eps)  # [n_out] bool

        # Replace pool = mature AND dead
        pool = eligible & dead_mask
        pool_count = pool.sum().astype(jnp.int32)

        # How many to replace: depend on pool size (not width)
        k_replace = jnp.where(
            pool_count > 0,
            jnp.maximum(jnp.ceil(rho * pool_count).astype(jnp.int32), 1),
            jnp.array(0, jnp.int32),
        )
        k_replace = jnp.minimum(k_replace, pool_count)

        # Rank within pool only (others get +inf so they sort last)
        score = jnp.where(pool, u_hat, jnp.inf)  # [n_out]
        sorted_idx = jnp.argsort(score)  # [n_out]

        # Fixed-shape replace mask (no dynamic slicing)
        take_sorted = (jnp.arange(n_out) < k_replace)  # [n_out] bool
        replace_mask = jnp.zeros((n_out,), dtype=bool).at[sorted_idx].set(take_sorted)

        # Apply replacement
        k_out, k_reinit = jax.random.split(k_in)
        W_new_full = lecun_reinit(k_reinit, W_in.shape)

        W_in_new = jnp.where(replace_mask[None, :], W_new_full, W_in)
        W_out_new = jnp.where(replace_mask[:, None], 0.0, W_out)

        # Reset CBP stats for replaced units
        u_new = jnp.where(replace_mask, 0.0, u)
        f_new = jnp.where(replace_mask, 0.0, f)
        a_new = jnp.where(replace_mask, 0.0, a)
        st_new = type(st)(u=u_new, f=f_new, a=a_new)

        replaced_count = replace_mask.sum().astype(jnp.int32)

        return (
            k_out,
            W_in_new,
            W_out_new,
            st_new,
            replace_mask,
            eligible,
            dead_mask,  # <-- new (optional, for logging)
            jnp.asarray(rho * pool_count),  # <-- changed: was rho*n_out
            k_replace,
            replaced_count,
            pool_count,  # <-- new (optional, for logging)
        )

        # --- Layer 0: rep -> hidden ---


    W_rep = p["rep"]["kernel"]  # [enc_out, feature_dim] (columns = rep units)
    W_hid = p["hidden"]["kernel"]  # [feature_dim, hidden_dim] (rows = rep units)
    rep_act = hs[0]  # [B, feature_dim]

    (
        k,
        W_rep,
        W_hid,
        st0,
        replace_mask0,
        eligible0,
        dead_mask0,
        k_replace_float0,
        k_replace_int0,
        replaced_rep,
        pool_count0,
    ) = process_layer(k, cbp.layers[0], rep_act, W_rep, W_hid)

    p["rep"]["kernel"] = W_rep
    p["hidden"]["kernel"] = W_hid

    # --- Layer 1: hidden -> q ---
    W_hid2 = p["hidden"]["kernel"]  # updated hidden kernel
    W_q = p["q"]["kernel"]  # [hidden_dim, num_actions] (rows = hidden units)
    hid_act = hs[1]  # [B, hidden_dim]

    (
        k,
        W_hid2,
        W_q,
        st1,
        replace_mask1,
        eligible1,
        dead_mask1,
        k_replace_float1,
        k_replace_int1,
        replaced_hid,
        pool_count1,
    ) = process_layer(k, cbp.layers[1], hid_act, W_hid2, W_q)

    p["hidden"]["kernel"] = W_hid2
    p["q"]["kernel"] = W_q

    total = replaced_rep + replaced_hid

    # params_out = FrozenDict(p)
    cbp_out = type(cbp)(layers=(st0, st1))  # CBPState

    # Return some debug signals (keep static shapes; masks are fixed [width])
    debug = {
        "replaced_rep": replaced_rep,
        "replaced_hid": replaced_hid,
        "replaced_total": total,
        "eligible_rep": eligible0.sum().astype(jnp.int32),
        "eligible_hid": eligible1.sum().astype(jnp.int32),
        "k_replace_rep": k_replace_int0,
        "k_replace_hid": k_replace_int1,
        "rho_n_out_rep": k_replace_float0,
        "rho_n_out_hid": k_replace_float1,
        "dead_rep": dead_mask0.sum().astype(jnp.int32),
        "dead_hid": dead_mask1.sum().astype(jnp.int32),
        "pool_count_rep": pool_count0,
        "pool_count_hid": pool_count1,
    }

    return k, p, cbp_out, debug