@testing.parameterize(*(testing.product({
    'batch': [1, 1, 0],
    'dtype': [numpy.float32],
}) + testing.product({
    'batch': [1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
})))
@testing.fix_random()
class TestLSTM1(TestLSTM):

    def setUp(self):
        hidden_shape = (1, 1, 1)
        x_shape = (self.batch, 4, 1)
        y_shape = (self.batch, 1, 1)
        self.c_prev = numpy.random.uniform(
            -1, 1, hidden_shape).astype(self.dtype)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        self.gc = numpy.random.uniform(-1, 1, hidden_shape).astype(self.dtype)
        self.gh = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)

        self.ggc = numpy.random.uniform(-1, 1, hidden_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        self.check_forward_options = {}
        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-2}
            self.check_backward_options = {'atol': 5e-3, 'rtol': 5e-2}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, c_prev_data, x_data):
        c_prev = chainer.Variable(c_prev_data)
        x = chainer.Variable(x_data)
        c, h = functions.lstm(c_prev, x)
        self.assertEqual(c.data.dtype, self.dtype)
        self.assertEqual(h.data.dtype, self.dtype)
        batch = len(x_data)

        # Compute expected out
        a_in = self.x[:, [0]]
        i_in = self.x[:, [1]]
        f_in = self.x[:, [2]]
        o_in = self.x[:, [3]]

        c_expect = _sigmoid(i_in) * numpy.tanh(a_in) + \
            _sigmoid(f_in) * self.c_prev[:batch]
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)

        testing.assert_allclose(
            c_expect, c.data[:batch], **self.check_forward_options)
        testing.assert_allclose(
            h_expect, h.data, **self.check_forward_options)
        testing.assert_allclose(
            c_prev_data[batch:], c.data[batch:], **self.check_forward_options)
