import ivy
import ivy.functional.frontends.paddle as paddle_frontend


class Tensor:
    def __init__(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    def __repr__(self):
        return (
            str(self._ivy_array.__repr__())
            .replace("ivy.array", "ivy.frontends.paddle.Tensor")
            .replace("dev", "place")
        )

    # Properties #
    # ---------- #
    @property
    def ivy_array(self):
        return self._ivy_array

    @property
    def place(self):
        return ivy.dev(self._ivy_array)

    @property
    def dtype(self):
        return self._ivy_array.dtype

    @property
    def shape(self):
        return self._ivy_array.shape

    # Setters #
    # --------#
    @ivy_array.setter
    def ivy_array(self, array):
        self._ivy_array = (
            ivy.array(array) if not isinstance(array, ivy.Array) else array
        )

    # Implement methods

    def __getitem__(self, query):
        ret = ivy.get_item(self._ivy_array, query)
        return paddle_frontend.Tensor(ivy.array(ret, dtype=ivy.dtype(ret), copy=False))
