import numpy as np
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

LEARNING_OFFSET = [1.1, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
BATCH_SIZE = [2, 4, 8, 16, 32, 64, 128, 256]
NUM_OF_TOPICS = np.arange(1,65).astype(int).tolist()


def lda_binary_decoder(X):
    doc_topic_prior = X[0]
    topic_word_prior = X[1]
    learning_decay = X[2]
    log_mean_change_tol = X[3]
    mean_change_tol = np.power(10, log_mean_change_tol)
    
    binary_code = X[4:]
    binary_code = (binary_code>=0.5).astype(int)
    #print(binary_code)
    
    bits_learning_offset = 3
    bits_batch_size = 3
    bits_n_components = 6
    
    curr = 0

    binary_learning_offset = ''.join([str(x) for x in binary_code[curr: curr+bits_learning_offset].tolist()])
    learning_offset = LEARNING_OFFSET[int(binary_learning_offset, 2)]
    curr += bits_learning_offset
    #print(learning_offset)
    
    binary_batch_size = ''.join([str(x) for x in binary_code[curr: curr+bits_batch_size].tolist()])
    batch_size = BATCH_SIZE[int(binary_batch_size, 2)]
    curr += bits_batch_size
    #print(batch_size)
    
    binary_n_components = ''.join([str(x) for x in binary_code[curr: curr+bits_n_components].tolist()])
    n_components = NUM_OF_TOPICS[int(binary_n_components, 2)]
    curr += bits_n_components
    #print(n_components)
    
    lda_config = {}
    lda_config['doc_topic_prior'] = doc_topic_prior
    lda_config['topic_word_prior'] = topic_word_prior
    lda_config['learning_decay'] = learning_decay
    lda_config['mean_change_tol'] = mean_change_tol
    lda_config['learning_offset'] = learning_offset
    lda_config['batch_size'] = batch_size
    lda_config['n_components'] = n_components
    
    return lda_config

def lda_binary_decoder_v2(X):
    
    if X.ndim == 2:
        X = np.squeeze(X)
    #
    
    doc_topic_prior = X[0]
    topic_word_prior = X[1]
    learning_decay = X[2]
    log_mean_change_tol = X[3]
    mean_change_tol = np.power(10, log_mean_change_tol)
    
    binary_code = X[4:]
    binary_code = (binary_code>=0.5).astype(int)
    #print(binary_code)
    
    bits_learning_offset = 3
    bits_batch_size = 3
    bits_n_components = 6
    
    curr = 0

    binary_learning_offset = ''.join([str(x) for x in binary_code[curr: curr+bits_learning_offset].tolist()])
    learning_offset = LEARNING_OFFSET[int(binary_learning_offset, 2)]
    curr += bits_learning_offset
    #print(learning_offset)
    
    binary_batch_size = ''.join([str(x) for x in binary_code[curr: curr+bits_batch_size].tolist()])
    batch_size = BATCH_SIZE[int(binary_batch_size, 2)]
    curr += bits_batch_size
    #print(batch_size)
    
    binary_n_components = ''.join([str(x) for x in binary_code[curr: curr+bits_n_components].tolist()])
    n_components = NUM_OF_TOPICS[int(binary_n_components, 2)]
    curr += bits_n_components
    #print(n_components)
    
    lda_config = {}
    lda_config['doc_topic_prior'] = doc_topic_prior
    lda_config['topic_word_prior'] = topic_word_prior
    lda_config['learning_decay'] = learning_decay
    lda_config['mean_change_tol'] = mean_change_tol
    lda_config['learning_offset'] = learning_offset
    lda_config['batch_size'] = batch_size
    lda_config['n_components'] = n_components
    
    return lda_config
        
def eval_lda_performance(domain, binary_config, max_epochs, mode, device):
    if binary_config.ndim == 2:
        binary_config = binary_config.squeeze()
        
    lda_config = lda_binary_decoder(binary_config)
    
    fixed_fidelity = [1,10,50]
    
    if mode == 'query':
        lda_model = LatentDirichletAllocation(
            max_iter=max_epochs, 
            learning_method='online',
            n_components=lda_config['n_components'], 
            doc_topic_prior=lda_config['doc_topic_prior'],
            topic_word_prior=lda_config['topic_word_prior'],
            learning_decay=lda_config['learning_decay'],
            learning_offset=lda_config['learning_offset'],
            batch_size=lda_config['batch_size'],
            mean_change_tol=lda_config['mean_change_tol'],
            random_state=0,
            n_jobs=8,
        )
        
        t0 = time()
        lda_model.fit(domain.tf_train)
        #print("done in %0.3fs." % (time() - t0))
        score = domain.metric(lda_model, device, score_type='perplexity')
        return score
    elif mode == 'generate':
        hist_scores = []
        for epochs in range(1,max_epochs+1):
            if epochs in fixed_fidelity:
                lda_model = LatentDirichletAllocation(
                    max_iter=epochs, 
                    learning_method='online',
                    n_components=lda_config['n_components'], 
                    doc_topic_prior=lda_config['doc_topic_prior'],
                    topic_word_prior=lda_config['topic_word_prior'],
                    learning_decay=lda_config['learning_decay'],
                    learning_offset=lda_config['learning_offset'],
                    batch_size=lda_config['batch_size'],
                    mean_change_tol=lda_config['mean_change_tol'],
                    random_state=0,
                    n_jobs=8,
                )

                t0 = time()
                lda_model.fit(domain.tf_train)
                #print("done in %0.3fs." % (time() - t0))
                score = domain.metric(lda_model, device, score_type='perplexity')
                hist_scores.append(score)
           #
        #
        return np.array(hist_scores)
    
    