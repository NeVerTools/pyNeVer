import numpy as np
from pynever.strategies.bp.bounds import HyperRectangleBounds

DEBUG = False


class PropertyFormatConverter:
    """
       A class used for converting a NeverProperty in format Cx<=d into two vectors: a lower_bound_vector and an
       upper_bound_vector.

       Attributes
       ----------
       coeff : Tensor
           The representation of matrix C
       bias : Tensor
           The representation of vector d
       """

    def __init__(self, property=None):
        if property is not None:
            self.coeff = property.in_coef_mat
            self.bias = property.in_bias_mat
        else:
            self.input_test()

        self.num_vars = self.coeff.shape[1]

        self.check_input_validity()

        self.lower_bound_vector = np.empty(self.num_vars, dtype=object)
        self.lower_bound_vector.fill(None)
        self.upper_bound_vector = np.empty(self.num_vars, dtype=object)
        self.upper_bound_vector.fill(None)

        self.get_vectors()

    def check_input_validity(self):
        """
        This code checks if every input variable has its own lower and upper value set, otherwise il closes the program
        """
        assert self.coeff.shape[0] == (2 * self.coeff.shape[1]), "Wrong property format: not convertible"
        assert self.coeff.shape[0] == self.bias.shape[0], "Wrong property format: not convertible"

        # Check that for each row in self.coeff matrix there is only one 1 or one -1

        for row in self.coeff:
            check = (all(x == 0 or x == -1 or x == 1 for x in row) and \
                     (np.count_nonzero(row == 1) + np.count_nonzero(row == -1) == 1))
            assert check, "Wrong property format: not convertible"

    def get_vectors(self) -> HyperRectangleBounds:

        for index, row in enumerate(self.coeff):
            index_of_one = np.where(row == 1)[0]
            index_of_minus_one = np.where(row == -1)[0]

            if np.size(index_of_one) == 1:
                self.upper_bound_vector[index_of_one[0]] = self.bias[index, 0]

            if np.size(index_of_minus_one) == 1:
                self.lower_bound_vector[index_of_minus_one[0]] = - self.bias[index, 0]

        # check that all elements of self.lower_bound_vector are lower than the related elements
        # of self.upper_bound_vector
        assert (all(self.lower_bound_vector <= self.upper_bound_vector)), "Wrong property format: not convertible"

        if DEBUG:
            print("lower_bound_vector: ", self.lower_bound_vector)
            print("upper_bound_vector: ", self.upper_bound_vector)

        return HyperRectangleBounds(self.lower_bound_vector, self.upper_bound_vector)
