import math

class CowMaskRegularizer:
  """CowMask regularizer."""

  def __init__(self, backg_noise_std, mask_prob, cow_sigma_range,
               cow_prop_range):
    self.backg_noise_std = backg_noise_std
    self.mask_prob = mask_prob
    self.cow_sigma_range = cow_sigma_range
    self.cow_prop_range = cow_prop_range
    self.log_sigma_range = (math.log(cow_sigma_range[0]),
                            math.log(cow_sigma_range[1]))
    self.max_sigma = cow_sigma_range[1]

  def perturb_sample(self, image_batch, rng_key):
    mask_size = image_batch.shape[1:3]
    prob_key, cow_key, noise_key = jax.random.split(rng_key, num=3)
    masks = cow_mask.cow_masks(
        len(image_batch), mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, cow_key)
    if self.mask_prob < 1.0:
      b = jax.random.bernoulli(prob_key, self.mask_prob,
                               shape=(len(image_batch), 1, 1, 1))
      b = b.astype(jnp.float32)
      masks = 1.0 + (masks - 1.0) * b
    if self.backg_noise_std > 0.0:
      noise = jax.random.normal(noise_key, image_batch.shape) * \
          self.backg_noise_std
      return image_batch * masks + noise * (1.0 - masks)
    else:
      return image_batch * masks

  def mix_images(self, image0_batch, image1_batch, rng_key):
    n_samples = len(image0_batch)
    mask_size = image0_batch.shape[1:3]
    masks = cow_mask.cow_masks(
        n_samples, mask_size, self.log_sigma_range, self.max_sigma,
        self.cow_prop_range, rng_key)
    blend_factors = masks.mean(axis=(1, 2, 3))
    image_batch = image0_batch + (image1_batch - image0_batch) * masks
    return image_batch, blend_factors