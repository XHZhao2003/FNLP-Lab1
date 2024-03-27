import numpy as np

class LogLinearModel:
    def __init__(self, feat_dim = 1024, label_dim = 4, alpha = 1e-4, beta = 0.1, weight_decay=1) -> None:
        self.feat_dim = feat_dim
        self.label_dim = label_dim
        self.alpha = alpha
        self.weights = np.random.rand(feat_dim, label_dim)        
        self.beta = beta
        self.gamma = weight_decay
        
        self.decayed_lr = self.alpha
                    
    def softmax(self, vector):
        score = np.exp(vector)
        return score / np.sum(score)
    
    def update(self, batch, label):
        '''
            label =         vector (batchsize)
            batch =         matrix (batchsize, feat_dim)
            weights =       matrix (feat_dim, label_dim)
            result =        matrix (batchsize, label_dim)
            pred =          matrix (batchsize, label_dim)
        '''
        result = np.dot(batch, self.weights)    
        pred = np.array([self.softmax(x) for x in result])
        batchsize = batch.shape[0]
        
        # compute batch loss
        loss = []
        for i in range(batchsize):
            loss.append(- np.log(pred[i][label[i]]))
        loss = sum(loss) / batchsize
        reg_loss = 0.5 * self.beta * np.linalg.norm(self.weights.flatten())
        loss += reg_loss
        
        # compute negative gradient
        neg_grad = np.zeros((self.feat_dim, self.label_dim))
        for index in range(batchsize):
            for i in range(self.feat_dim):
                for r in range(self.label_dim):
                    neg_grad[i][r] -= batch[index][i] * pred[index][r]
            truth = label[index]
            for i in range(self.feat_dim):
                neg_grad[i][truth] += batch[index][i]
        for i in range(self.feat_dim):
            for r in range(self.label_dim):
                neg_grad[i][r] = neg_grad[i][r] / batchsize
                neg_grad[i][r] -= self.weights[i][r] * self.beta
        
        # update weights
        step = np.multiply(neg_grad, np.array(self.decayed_lr))
        self.weights += step
                
        return loss, reg_loss
    
    def predict(self, batch):
        result = np.dot(batch, self.weights)    
        return np.array([self.softmax(x) for x in result])

    def save(self, path):
        np.savetxt(path, self.weights)
    
    def load(self, path):
        self.weights = np.loadtxt(path)
        