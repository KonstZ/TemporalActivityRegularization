
from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf

class TemporalActivityRegularizer(Layer):
	'''
		TemporalActivityRegularizer - a layer to penalyze current variation of activations with respect to per-sample average
		Temporal Ensembling for Semi-Supervised Learning https://arxiv.org/abs/1610.02242
		Requires additional input with sample indexes (zero index is masked out)

		# Example
			...
			preds = Dense(len(labels_index))(x)
			preds = TemporalActivityRegularizer(weight = 0.1, moment=0.9, warm_up = 1e3, cool_down = 1e7, max_items = 15000)([preds,ids_input])
	
		# Params:
			weight: weight in the total loss
			moment: for updating history
			max_items: restriction for the number of samples
			warm_up: number of inintial iterations with reduce weight
			coold_down: number of iterations after which to reduce weight
	'''
	def calc_loss(self, x):
		activations = x[0]
		samples = x[1]
		mask = K.cast(K.not_equal(samples, 0), K.floatx()) * K.cast(K.lesser(samples, self.max_items), K.floatx())
		masked_samples = K.flatten(K.cast(K.minimum(samples, int(self.max_items)), 'int32'))
		old_activations = K.gather(self.history, masked_samples)
		diff_activations = (old_activations - activations) * mask
		warm_up = self.warm_up * self.iterations
		cool_down = self.cool_down * self.iterations
		loss = self.weight * K.mean(K.square(diff_activations)) * warm_up / (1 + warm_up) / (1 + cool_down)
		return loss

	def calc_update(self, x):
		activations = x[0]
		samples = x[1]
		mask = K.cast(K.not_equal(samples, 0), K.floatx()) * K.cast(K.lesser(samples, self.max_items), K.floatx())
		masked_samples = K.flatten(K.cast(K.minimum(samples, int(self.max_items)), 'int32'))
		old_activations = K.gather(self.history, masked_samples)
		diff_activations = (old_activations - activations) * mask
		return tf.scatter_sub(self.history, masked_samples, (1 - self.moment) * diff_activations)

	def build(self, input_shape):
		history_shape = (self.max_items + 1,) + input_shape[0][1:]
		self.history = self.add_weight(history_shape, initializer='zero',
			name='{}_history'.format(self.name), trainable=False)
		self.loss_added = False

	def get_output_shape_for(self, input_shape):
			return input_shape[0]

	def call(self, x, mask=None):
		if not self.loss_added:
			self.add_loss(self.calc_loss(x))
			self.add_update(self.calc_update(x))
			self.add_update(K.update_add(self.iterations, 1))
			self.loss_added = True
		return x[0]

	def get_config(self):
		config = {'moment': self.moment,
				'weight': self.weight,
				'warm_up': self.warm_up,
				'cool_down': self.cool_down,
				'max_items': self.max_items}
		base_config = super(TemporalActivityRegularizer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def __init__(self, weight = 0.1, moment = 0.5, warm_up = 1000, cool_down = 100000, max_items = 100000, **kwargs):
		self.weight = weight
		self.moment = moment
		self.max_items = max_items
		self.iterations = K.variable(0.)
		self.warm_up = 1.0 / warm_up
		self.cool_down = 1.0 / cool_down
		super(TemporalActivityRegularizer, self).__init__(**kwargs)


