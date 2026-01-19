"""
LESSON 1: JAX/FLAX ENGINEERING PATTERNS
---------------------------------------
Author: CDY
Focus: Memorizing the "Translation Layer" between Math and Code.

KEY CONCEPTS TO MEMORIZE:
1. Generators: Yielding data lazily (essential for massive bio-datasets).
2. Flax Module: Using @nn.compact for inline layer definition.
3. TrainState: The container that holds params + optimizer.
4. JIT & Grad: The strict data flow required for XLA compilation.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np  # For data simulation
from typing import Iterator

# ==========================================
# PATTERN 1: THE DATA GENERATOR (Lazy Loading)
# ==========================================
# MEMORIZE: Use 'yield' to stream data instead of loading it all into RAM.
def data_stream(n_samples: int = 100) -> Iterator[dict]:
    """Simulate a massive biological dataset."""
    for i in range(n_samples):
        x = np.random.uniform(0, 1)
        y = 2 * x + 1 + np.random.normal(0, 0.1)  # y = 2x + 1 + noise
        
        # Yield one sample at a time
        yield {"x": np.array([x]), "y": np.array([y])}

# ==========================================
# PATTERN 2: THE FLAX MODEL (Compact)
# ==========================================
# MEMORIZE: @nn.compact allows you to define .__call__ directly without setup().
class LinearModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # MEMORIZE: Flax uses 'Dense', not 'Linear'.
        return nn.Dense(features=1)(x)

# ==========================================
# PATTERN 3: THE STATE CONTAINER
# ==========================================
def create_train_state(rng, learning_rate=0.1):
    """Initializes the model and the optimizer."""
    model = LinearModel()
    
    # MEMORIZE: .init() creates the parameters. It needs a dummy input shape.
    params = model.init(rng, jnp.ones([1, 1]))["params"]
    
    # MEMORIZE: Optax handles the optimizer logic (SGD/Adam).
    tx = optax.adam(learning_rate)
    
    # MEMORIZE: TrainState packages the Model Logic (apply_fn), Params, and Optimizer.
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

# ==========================================
# PATTERN 4: LOSS FUNCTION WITH AUXILIARY DATA
# ==========================================
# MEMORIZE: If you want to see predictions during training, return (loss, preds).
# But if you do this, you MUST set has_aux=True in the gradient step.
def calculate_loss(params, x, y, apply_fn):
    # Run forward pass
    predictions = apply_fn({"params": params}, x)
    
    # MSE Loss
    loss = jnp.mean((predictions - y) ** 2)
    
    # Return TUPLE: (Metric to minimize, Extra Info)
    return loss, predictions

# ==========================================
# PATTERN 5: THE TRAINING STEP (The Core)
# ==========================================
# MEMORIZE: @jax.jit compiles this function. All inputs must be JAX arrays or marked static.
@jax.jit
def train_step(state, x, y):
    """
    1. Calculate Loss & Grads
    2. Update Params
    3. Return New State
    """
    
    # MEMORIZE: Partial application for the loss function.
    # We pass 'state.apply_fn' (static logic) into the loss calculator.
    def loss_fn(params):
        return calculate_loss(params, x, y, state.apply_fn)
    
    # MEMORIZE: value_and_grad(..., has_aux=True)
    # returns: ((loss_val, aux_val), gradients)
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # MEMORIZE: state.apply_gradients() is the only way to update weights in Flax.
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss

# ==========================================
# EXECUTION (Putting it all together)
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng)
    
    # 2. Fake Data Batch (Normally comes from generator)
    # In JAX, we usually batch data explicitly.
    x_batch = jnp.array([[0.1], [0.5], [0.9]])
    y_batch = 2 * x_batch + 1  # Target
    
    print("Starting Training...")
    
    # 3. Training Loop
    for epoch in range(101):
        # CRITICAL: You must capture the *returned* state.
        # State is immutable; train_step returns a NEW state object.
        state, loss = train_step(state, x_batch, y_batch)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            
    # 4. Final Verification
    print("\nOptimization Complete.")
    final_pred = state.apply_fn({"params": state.params}, jnp.array([[0.5]]))
    print(f"Target for input 0.5: 2.0")
    print(f"Model prediction:     {final_pred[0][0]:.4f}")