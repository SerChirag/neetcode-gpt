import numpy as np
from numpy.typing import NDArray


class Solution:

    def logsumexp(self, z):
        c = z.max()
        return np.log(np.sum(np.exp(z - c)))

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        c = z.max()
        softmax = np.exp(z - c - self.logsumexp(z))
        return np.round(softmax,4)
