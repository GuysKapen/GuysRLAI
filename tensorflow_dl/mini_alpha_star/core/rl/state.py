import time


class MsState(object):
    def __init__(self, entity_state=None, statistical_state=None,
                 map_state=None):
        super(MsState, self).__init__()
        self.entity_state = entity_state  # shape b x entity_size x embedding_size

        # or scalar state
        self.statistical_state = statistical_state  # list of tensors

        # or spatial state
        self.map_state = map_state

        self._shape = None

    def _get_shape(self):
        shape1 = str(self.entity_state.shape)
        shape2 = ["%s" % str(s.shape) for s in self.statistical_state]
        shape3 = str(self.map_state.shape)

        self.shape1 = f'\nentity_state: {shape1};'
        self.shape2 = f'\nstatistical_state: {shape2};'
        self.shape3 = f'\nmap_state: {shape3}.'

        self._shape = self.shape1 + self.shape2 + self.shape3

    def to_list(self):
        return [self.entity_state, self.statistical_state, self.map_state]

    @property
    def shape(self):
        if self._shape is None:
            self._get_shape()
        return self._shape

    def __str__(self):
        if self._shape is None:
            self._get_shape()

        return self._shape

