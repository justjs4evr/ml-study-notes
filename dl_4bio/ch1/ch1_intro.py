import jax
import jax.numpy as jnp

def compute_ten_power_sum(arr: jax.Array) -> float:
    """raise values to the power of 10 and then sum."""
    return jnp.sum(arr**10)

arr = jnp.array([1,2,3,4,5]) #how to create a jax array using jnp, similar to numpy
###print(compute_ten_power_sum(arr))

jitted_compute_ten_power_sum = jax.jit(compute_ten_power_sum)
###print(jitted_compute_ten_power_sum(arr))

###or, via the python syntactic sugar, we can apply it when defining the function with the @jax.jit decorator
@jax.jit
def compute_ten_power_sum(arr: jax.Array) -> float:
    return jnp.sum(arr**10)
###print(compute_ten_power_sum(arr))


#preconfiguring JAX jit with partial method

from functools import partial

def scale(x, scaling_factor):
    return x*scaling_factor

#cfreating a new func 'scale_by_10' where scaling factor is fixed to 10
scale_by_10 = partial(scale,scaling_factor=10)
###print(scale_by_10(5))  #should print 50

@partial(jax.jit, static_argnums=(0,))
def summarize(average_method: str, x: jax.Array) -> float:
    if average_method == "mean":
        return jnp.mean(x)
    elif average_method == "median":
        return jnp.median(x)
    else:
        raise ValueError(f"Unsupported average type: {average_method}")
    

data_array = jnp.array([1.0, 2.0, 100.0])

#jax compiles one version of summarize for averagemethod mean
##print(f"Mean: {summarize('mean', data_array)}")

#jax compiles another version of summarize for averagemethod median
##print(f"Median: {summarize('median', data_array)}")

#calling with mean again uses the cashed compiled version
##print(f"Mean: {summarize('mean', data_array)}")

##closures: remembers the environment
def outer_function(x: float) -> callable:
    def inner_function(y: float) -> float:
        return x + y # inner_function "closes over" x
    return inner_function

add_five = outer_function(5)
result = add_five(10) # y is 10
#print(f"closure result: {result}") # should be 15 


# another example by me
def  power_factory(x: float) -> callable:
    def inner_function(y: float) -> float:
        return x**y
    return inner_function

power_ten = power_factory(10)
result = power_ten(2) #y is 2
#print(f"closure result2: {result}") #should be 100

### generators: functions that allow you to iterate over data lazily, yielding one item at a time

#here is a simple gernator function:
from typing import Iterator

def data_generator() -> Iterator[dict]:
    """yield data samples with features and labels"""
    for i in range(5):
        yield {"feature":i, "label": i%2}

#example usage
generator = data_generator()
next(generator)

#we will be working with tensorflow dataset - since JAX doesnt include native data-loading library
import tensorflow as tf
import numpy as np
features = np.array([1,2,3,4,5])
labels = np.array([0,1,0,1,0])

dataset = tf.data.Dataset.from_tensor_slices((features, labels))

batched_dataset = dataset.batch(2, drop_remainder=True)


#create a dataset (ds) iterator and retrieve the first batch using next()
#ds = iter(batched_dataset)
#next(ds)


#anatomy of a training loop using jax / flax
#import jax
#import jax.numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn

#in jax, randomness is handled explicitly by passing a random key
#we create a key here to seed the random number generator
rng = jax.random.PRNGKey(42)

#toy data:
rng, rng_data, rng_noise = jax.random.split(rng, 3)
x_data = jax.random.uniform(rng_data, shape=(100,1))

# add gaussian noise
noise = 0.1 * jax.random.normal(rng_noise, shape=(100,1))

#define target: y = 2x + 1 + noise
y_data = 2 * x_data + 1 + noise

#visualise the noisy linear relationship
plt.scatter(x_data, y_data)
plt.xlabel("x")
plt.ylabel("y")
plt.title("toy dataset: y = 2x + 1 + noise")
#plt.show() #Figure 1-1 random


#lets define a model using @nn.compact

class LinearModel(nn.Module):
    @nn.compact ## allows __call__ directly defined layers, rather than in setup() method of the class
    def __call__(self, x):
        return nn.Dense(features=1)(x) 
        #applies a single dense(fully connected) layer with 1 output neuron
        #that is, it computes y = xW + b, where the output has dim 1.
    
model = LinearModel()


rng = jax.random.PRNGKey(42)
variables = model.init(rng, jnp.ones([1,1]))
##print_short_dict(variables)

#creating a training state = container that packages together all you need: parameters, optimizer, apply-model
import optax
from flax.training import train_state

#define an optimizer = here we use Adam with a learning rate of 1.0
# (Note: in most real settings you'd use a smaller learning rate like 1e-3).
tx = optax.adam(1.0)

#create the training state
state = train_state.TrainState.create(
    apply_fn = model.apply, #forward pass func
    params=variables["params"], # the initialized mode parameters
    tx=tx, # the optimizer
)


#define loss function
def calculate_loss(params, x, y):
    #run a forward pass of the model to get predictions
    predictions = model.apply({"params":params}, x)
    #compute MSE Loss
    return jnp.mean((predictions - y) ** 2)

loss = calculate_loss(variables["params"], x_data, y_data)
#print(f"loss: {loss:.4f}")


#defining a training step

"""@jax.jit
def train_step(state, x, y):
    #compute the loss and its gradients with respect to the parameters
    loss, grads = jax.value_and_grad(calculate_loss)(state.params, x,y)
    #apply graient updates
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

""" 
# we just use jax.value_and_grad & closure func
"""@jax.jit
def train_step(state,x,y):
    def calculate_loss(params):
        #state, x, y are not part of the function signature but are accessed!
        predictions = state.apply_fn({"params":params}, x)
        return jnp.mean((predictions - y) ** 2)
    
    loss, grads = jax.value_and_grad(calculate_loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss"""

#handling auxiliary outputs in the loss function
@jax.jit
def train_step(state, x, y):
    def calculate_loss(params):
        predictions = state.apply_fn({"params": params}, x)
        loss = jnp.mean((predictions - y) ** 2)
        return loss, predictions
        
    (loss, predictions), grads = jax.value_and_grad(calculate_loss, has_aux = True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, predictions)    
 ## return both loss and prediction (aux info)

#defining the traing loop
num_epochs = 150 # number of full passes through the training data

for epoch in range(num_epochs):
    state, (loss, _) = train_step(state, x_data, y_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


#we can test how well the model has learned the underlying pattern by comparing its predictions to the true target values

#generate test data
x_test = jnp.linspace(0, 1, 10).reshape(-1,1)
y_test = 2*x_test + 1 # ground truth = linear function without noise.

#get model predictions
y_pred = state.apply_fn({"params": state.params}, x_test)
plt.clf()
plt.scatter(x_test, y_test, label="true values")
plt.plot(x_test, y_pred, label="model predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("linear model pred vs true relationship")
plt.show()