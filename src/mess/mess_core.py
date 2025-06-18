import numpy as np

from scipy.spatial.distance import pdist, squareform


class MESS():

    def __init__(self, sequence, analysis_type='state'):
        self.sequence = sequence
        self.analysis_type = analysis_type
        self.state_matrix = None
        self.recurrence_matrix = None
        self.sampled_sequence_elements = None
        
        if analysis_type == 'state':
            self.state_matrix = pdist(self.sequence.T, metric='euclidean')
            self.state_matrix = squareform(self.state_matrix)
            self.state_matrix /= self.state_matrix.max()

        elif analysis_type == 'energy':
            energy_states = np.diff(self.sequence, axis=1)
            energy_states = np.abs(energy_states)
            energy_states = np.linalg.norm(energy_states, axis=0)**2

            self.state_matrix = np.abs(energy_states[:, None] - energy_states[None, :])
            self.state_matrix /= self.state_matrix.max()

    def sampling(self, eps=0.01):
        self.recurrence_matrix = self.state_matrix < eps
        indices = self.__maximum_entropy_snapshot_sampling()

        if self.analysis_type == 'energy':
            indices = np.append(indices, False)
            
        self.sampled_sequence_elements = self.sequence[:, indices]

        return self.sampled_sequence_elements


    def __maximum_entropy_snapshot_sampling(self):
        n = self.recurrence_matrix.shape[1]
        idx = np.zeros(n, dtype=bool)
        idx[0] = True

        v = 1
        k = 1

        for cnt in range(n-1):
            d = 2 * np.sum(self.recurrence_matrix[cnt+1, idx]) + 1

            if (d - (2*k + 1)*v) < 0:
                idx[cnt+1] = True
                v = (k*k*v + d) / ((k+1)**2)
                k += 1

        return idx

   

