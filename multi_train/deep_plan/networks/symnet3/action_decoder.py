from tensorflow.keras.layers import Layer
import tensorflow as tf

class ActionDecoder(Layer):
	def __init__(self, params): # Decoder dimension refers to the same param as Sankalp
		"""
		:param params:  type, activation
		"""
		super(ActionDecoder,self).__init__()
		self.ad_params = params
		self.type = params["type"]
		self.activation = self.get_activation(self.ad_params['activation'])
		self.use_ge = params["use_ge"]
		self.dropout_rate = params["dropout_rate"]

		self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
		if self.type == "symnet":
			self.layer1 = tf.keras.layers.Dense(units=self.ad_params["decoder_dim"], activation=self.activation)
			self.layer2 = tf.keras.layers.Dense(units=1)
		else:
			raise ValueError("networks.ActionDecoder: Invalid actiondecoder type: " + self.type)

	def get_activation(self, activation):
		if activation == "relu":
			return tf.nn.relu
		elif activation == "lrelu":
			return tf.nn.leaky_relu

	def call(self, inputs):
		"""
		:param inputs: [node_embed, global_embed] or [global_embed, None] for parametrized and non-parameterized nodes.
		Shape of both = [num_node, F]
		:return:
		"""
		node_embed, global_embed, training = inputs
		if self.use_ge and global_embed is not None:
			# Concat node and global embeddings
			# num_nodes = node_embed.shape[1]
			# global_embed = tf.reshape(tf.tile(global_embed, [1, num_nodes]), node_embed.shape)
			node_embed = tf.concat([node_embed, global_embed], axis=-1)

		# node_embed = self.dropout_layer(node_embed, training)
		if self.type == "symnet":
			node_embed = self.layer1(node_embed)
			node_embed = self.dropout_layer(node_embed, training)
			node_embed = self.layer2(node_embed)
			return node_embed

if __name__ == "__main__":
	actionDecoder = ActionDecoder(4,tf.nn.relu)
	print(actionDecoder.layer1.get_weights())