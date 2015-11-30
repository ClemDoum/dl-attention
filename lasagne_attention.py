import numpy as np
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import MergeLayer, Gate, InputLayer
from lasagne.utils import unroll_scan
from theano import tensor as T, theano


class GRUDecoder(MergeLayer):
    def __init__(self, incoming, num_units, num_unit_attention,
                 output_voca_size,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 W_att_enc=init.GlorotUniform(),
                 W_att_dec=init.GlorotUniform(),
                 W_att_out=init.Uniform(),
                 W_out=init.GlorotUniform(),
                 b_out=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)

        # Initialize parent layer
        super(GRUDecoder, self).__init__(incomings, **kwargs)

        self.output_voca_size = output_voca_size
        self.learn_init = learn_init
        self.num_units = num_units
        self.num_unit_attention = num_unit_attention
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = False

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
            hidden_update, 'hidden_update')

        # The attention parameters
        self.W_att_enc = self.add_param(
            W_att_enc, (self.num_units, self.num_unit_attention),
            name="W_att_enc")
        self.W_att_dec = self.add_param(
            W_att_dec, (self.num_units, self.num_unit_attention),
            name="W_att_dec")
        self.W_att_out = self.add_param(
            W_att_out, (self.num_unit_attention,), name="W_att_out")

        # Add the softmax for the output
        self.W_out = self.add_param(W_out,
                                    (self.num_units, self.output_voca_size),
                                    name="W_out")
        self.b_out = self.add_param(b_out, (self.output_voca_size,),
                                    name="b_out")

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def attention(self, encoder_output, hidden_previous, W_att_enc, W_att_dec,
                  W_att_out, mask=None):
        # Compute the alignment between the encoder output and the current state
        alpha = T.dot(encoder_output, W_att_enc)
        alpha += T.dot(hidden_previous, W_att_dec)[None, :, :]
        alpha = T.tanh(alpha)
        attention_vector = T.nnet.softmax(T.dot(alpha, W_att_out.T))
        # Compute the attention as the sum of the encoder output
        # weighted by the alignment weights
        return (attention_vector[:, :, None] * encoder_output).sum(axis=0)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        encoder_output = inputs[1]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)
            encoder_output = T.flatten(encoder_output, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        encoder_output = encoder_output.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, encoder_output, W_hid_stacked,
                 W_in_stacked, b_stacked, W_att_enc, W_att_dec, W_att_out,
                 W_out, b_out):
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping is not False:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate * hidden_update_hid
            if self.grad_clipping is not False:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous + updategate * hidden_update

            # # Add the attention
            hid += self.attention(encoder_output, hid_previous, W_att_enc,
                                  W_att_dec,
                                  W_att_out)

            # Compute the probas
            probs = T.nnet.softmax(T.dot(hid, W_out) + b_out)
            return [hid, probs]

        sequences = [input]
        step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [encoder_output, W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked, self.W_att_enc,
                         self.W_att_dec, self.W_att_out, self.W_out,
                         self.b_out, ]
        # theano.scan only allows for positional arguments, so when
        # self.precompute_input is True, we need to supply fake placeholder
        # arguments for the input weights and biases.
        else:
            non_seqs += [(), (), self.W_att_enc, self.W_att_dec,
                         self.W_att_out, self.W_out, self.b_out]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            out, _ = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            out, _ = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, None],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        # hid_out = hid_out[0].dimshuffle(1, 0, 2)
        s_out = out[1]

        # # if scan is backward reverse the output
        # if self.backwards:
        #     out = out[:, ::-1, :]

        return s_out


if __name__ == '__main__':
    floatX = "float32"
    intX = "int32"
    # Data
    batch_size = 10
    input_sentence_length = 88
    hidden_dim = 13
    attention_dim = 11
    embeddings_dim = 17
    output_voca_size = 134

    # n_decodesteps = 100
    # nb_passes = 3

    input_sentences = T.tensor3("input_sentences", dtype=floatX)
    encoder_output = T.tensor3("encoder_output", dtype=floatX)
    # output_sentences = T.tensor3("output_sentences", dtype=floatX)

    l_in = InputLayer(
        shape=(batch_size, input_sentence_length, embeddings_dim),
        input_var=input_sentences)
    layer = GRUDecoder(l_in, hidden_dim, attention_dim, output_voca_size)

    output_sentences = layer.get_output_for([input_sentences, encoder_output])
    fn = theano.function([input_sentences, encoder_output], output_sentences,
                         # mode='DebugMode',
                         on_unused_input='ignore')

    np_encoder_output = inputs = np.random.normal(
        size=(batch_size, input_sentence_length, hidden_dim)).astype(floatX)
    np_input_sentences = np.random.normal(
        size=(batch_size, input_sentence_length, embeddings_dim)).astype(
        floatX)
    np_output_sentences = fn(np_input_sentences, np_encoder_output)
    print np_output_sentences
    print np_output_sentences.shape
