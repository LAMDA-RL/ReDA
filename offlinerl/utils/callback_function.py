from http.client import PARTIAL_CONTENT


class EarlyStopCallback(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.reset()
        
    def reset(self):
        self.duration = 0
        self.best_score = None
        self.best_parameters = None
        
    def set(self, score, parameters):
        self.duration = 0
        self.best_score = score
        self.best_parameters = parameters
        
    def update(self, score, parameters):
        if self.best_score is None or score < self.best_score:
            self.set(score, parameters)
        else:
            self.duration += 1
        if self.duration >= self.patience:
            return False
        return True
    
    def get(self):
        return self.best_score, self.best_parameters