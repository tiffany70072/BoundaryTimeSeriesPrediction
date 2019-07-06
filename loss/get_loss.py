# Reference: Reproduce the Experienments in our paper: Deep Censored Learning of the Winning Price in the Real Time Bidding
#            https://github.com/wush978/deepcensor

import keras.backend as K
import tensorflow as tf
import numpy as np
from numpy import float32
from callback import EpochCoordinateDescent
import scipy

def get_loss_yes(X, wp_bp, batch_size = None):
    print('get loss')
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000}
    print('sigma =', K.get_value(param["sigma"]))

    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)
        
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        wp_loglik = - (wp_rss / (2 * sigma_square) + np.log(2 * np.pi * sigma_square) / 2) # 為什麼這一坨這麼長＝＝
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    
    def training_loss(y_true, y_pred):
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        from scipy.stats import norm
        
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            bp_loglikelihood = norm.logsf(bp, bp_pred, sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    
    return (param, training_loss, loglikelihood, callbacks, lambda x : x, lambda x : x, np.nanmean(wp))

def get_loss_mse(X, wp_bp, batch_size = None):
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]
    
    def mse_loss(y_true, y_pred):
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]

        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)

        #K.set_value(param["sigma"], 1.0)
        #bp_mse = tf.square(tf.math.minimum(wp_pred - bp, 0))
        #bp_mse = tf.square(tf.clip_by_value(bp - wp_pred, 0.0, 10000.0))
        bp_mse = tf.square(tf.clip_by_value(wp_pred - bp, -10000.0, 0.0))
        
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_mse = tf.square(wp_pred - wp_true)
        wp_mse = tf.scatter_nd(wp_idx, wp_mse, tf.shape(is_win))
        mse = tf.where(is_win, wp_mse, bp_mse)
        
        return mse
    
    def training_loss(y_true, y_pred):
        mse = mse_loss(y_true, y_pred)
        return K.mean(mse) 
    
    return (training_loss, None)

def get_loss_mse_sigma(X, wp_bp, batch_size = None): # two stage loss for cdf-mse
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]

    def mse_loss(y_true, y_pred):
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]

        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)

        # diff
        sigma = 35.
        # lower_mse = tf.square(tf.clip_by_value(wp_pred - (bp+sigma), -10000.0, 0.0))
        # upper_mse = tf.square(tf.clip_by_value(wp_pred - 3*(bp+sigma), 0.0, 10000.0))
        # ratio
        sigma = 2.
        lower_mse = tf.square(tf.clip_by_value(wp_pred - (bp*sigma*0.5 + bp*0.5), -10000.0, 0.0))
        upper_mse = tf.square(tf.clip_by_value(wp_pred - (bp*sigma*1.5 + bp*0.5), 0.0, 10000.0))
        # lower_mse = tf.square(tf.clip_by_value(wp_pred - (bp), -10000.0, 0.0))
        # upper_mse = tf.square(tf.clip_by_value(wp_pred - (bp*sigma*2 - bp), 0.0, 10000.0))
        
        bp_mse = lower_mse + upper_mse

        # two stage
        #original_mse = tf.square(tf.clip_by_value(wp_pred - bp, -10000.0, 0.0))
        #extra_mse = tf.square(wp_pred - bp*sigma)
        #bp_mse = original_mse + 0.1 * extra_mse
        
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_mse = tf.square(wp_pred - wp_true)
        wp_mse = tf.scatter_nd(wp_idx, wp_mse, tf.shape(is_win))
        mse = tf.where(is_win, wp_mse, bp_mse)
        
        return mse
    
    def training_loss(y_true, y_pred):
        mse = mse_loss(y_true, y_pred)
        return K.mean(mse) 
    
    return (training_loss, None)


def get_loss_diff(X, wp_bp, batch_size = None):
    #print('get loss')
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000,
            "diff": K.variable(400.0)}
            #"diff": K.variable(np.nanmean(wp) - np.nanmean(wp_bp[:, 1]))}
        
    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        diff = K.get_value(param["diff"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)

        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        bp_loglik = tf.clip_by_value(bp_loglik, 0.0, 0.5)

        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        wp_loglik = - (wp_rss / (2 * sigma_square) + np.log(2 * np.pi * sigma_square) / 2) # 為什麼這一坨這麼長＝＝
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    
    def training_loss(y_true, y_pred):
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        from scipy.stats import norm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            bp_loglikelihood = norm.logsf(bp, bp_pred, sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        #diff_value, diff_likelihood_min, info = fmin_l_bfgs_b(diff_loglikelihood, [K.get_value(param["diff"])], approx_grad = True, bounds = [(None, None)])
        
        K.set_value(param["sigma"], sigma_value[0])
        
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    return (param, training_loss, loglikelihood, callbacks, lambda x : x, lambda x : x, np.nanmean(wp))

def get_loss_cdf(X, wp_bp, batch_size = None):
    print('get loss')
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000}
    print('sigma =', K.get_value(param["sigma"]))

    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        wp_loglik = - (wp_rss / (2 * sigma_square) + np.log(2 * np.pi * sigma_square) / 2) # 為什麼這一坨這麼長＝＝
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    
    def training_loss(y_true, y_pred):
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        from scipy.stats import norm
        
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            bp_loglikelihood = norm.logsf(bp, bp_pred, sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    return (param, training_loss, loglikelihood, callbacks, lambda x : x, lambda x : x, np.nanmean(wp))


def get_loss_sigma(X, wp_bp, batch_size = None):
    print('get loss')
    if batch_size is None: batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(wp)), "ratio" : 1000}
    print('sigma =', K.get_value(param["sigma"]))

    def loglikelihood(y_true, y_pred):
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        sigma = K.get_value(param["sigma"])
        print('sigma1 =', sigma.shape) # float32

        sigma = sigma * wp_pred
        sigma_square = np.square(sigma)
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win)) # know wp only when is_win = True (bp = wp)
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        sigma_square_mask = tf.boolean_mask(sigma, is_win)
        #x = wp_rss / (2 * sigma_square_mask)
        #y = tf.log(2 * np.pi * sigma_square_mask) / 2
        #z = x + y
        
        wp_loglik = - (wp_rss / (2 * sigma_square_mask) + tf.log(2 * np.pi * sigma_square_mask) / 2) # 為什麼這一坨這麼長＝＝
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    
    def training_loss(y_true, y_pred):
        sr = -loglikelihood(y_true, y_pred)
        return K.mean(sr) * param["ratio"]
    
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        from scipy.stats import norm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = norm.logpdf(wp, wp_pred, sigma)
            bp_loglikelihood = norm.logsf(bp, bp_pred, sigma)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            sigma = (wp_pred * param["sigma"])
            lower = sigma * 0.1 
        from scipy.optimize import fmin_l_bfgs_b
        #sigma = [K.get_value(param["sigma"])] * K.get_value(tf.unstack(y_pred, num = 2, axis = 1)[0])
        sigma = (wp_pred * param["sigma"])
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, sigma, approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    
    return (param, training_loss, loglikelihood, None, lambda x : x, lambda x : x, np.nanmean(wp))

def get_loss_lognormal(X, wp_bp, batch_size = None):
    if batch_size is None:
        batch_size = X.shape[0]
    wp = wp_bp[:,0]
    param = {"sigma" : K.variable(np.nanstd(np.log(wp))), "ratio" : 1000}
    def loglikelihood(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.unstack(y_true, num = 2, axis = 1)[0]
        bp = tf.unstack(y_true, num = 2, axis = 1)[1]
        y_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        ds = tf.contrib.distributions
        dist = ds.TransformedDistribution(
          distribution=ds.Normal(loc=y_pred, scale=sigma),
          bijector=ds.bijectors.Exp())
        #dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win))
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(y_pred, is_win)
        wp_dist = ds.TransformedDistribution(
          distribution=ds.Normal(loc=wp_pred, scale=sigma),
          bijector=ds.bijectors.Exp())
        wp_loglik = wp_dist.log_prob(wp_true)
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return loglik
    def training_loss(y_true, y_pred):
        sigma = K.get_value(param["sigma"])
        sigma_square = np.square(sigma)
        wp_true = tf.log(tf.unstack(y_true, num = 2, axis = 1)[0])
        bp = tf.log(tf.unstack(y_true, num = 2, axis = 1)[1])
        wp_pred = tf.unstack(y_pred, num = 2, axis = 1)[0]
        dist = tf.distributions.Normal(loc=wp_pred, scale=sigma)
        bp_loglik = dist.log_survival_function(bp)#tf.zeros(shape = tf.shape(bp))
        is_not_win = tf.is_nan(wp_true)
        is_win = tf.logical_not(is_not_win)
        wp_idx = tf.to_int32(tf.where(is_win))
        wp_true = tf.boolean_mask(wp_true, is_win)
        wp_pred = tf.boolean_mask(wp_pred, is_win)
        wp_rss = tf.square(wp_pred - wp_true)
        wp_loglik = - (wp_rss / (2 * sigma_square) + np.log(2 * np.pi * sigma_square) / 2)
        wp_loglik = tf.scatter_nd(wp_idx, wp_loglik, tf.shape(is_win))
        loglik = tf.where(is_win, wp_loglik, bp_loglik)
        return K.mean(-loglik) * param["ratio"]
    def param_updater(y_true, y_pred, param, lower = None):
        wp = y_true[:,0]
        is_win = ~np.isnan(wp)
        wp = wp[is_win]
        bp = y_true[~is_win,1]
        wp_pred = y_pred[is_win,0]
        bp_pred = y_pred[~is_win,0]
        wp_pred_exp = np.exp(wp_pred)
        bp_pred_exp = np.exp(bp_pred)
        from scipy.stats import lognorm
        def sigma_loglikelihood(sigma):
            wp_loglikelihood = lognorm.logpdf(wp, s = sigma, scale = wp_pred_exp)
            bp_loglikelihood = lognorm.logsf(bp, s = sigma, scale = bp_pred_exp)
            return -(np.sum(wp_loglikelihood) + np.sum(bp_loglikelihood))
        if lower is None:
            lower = K.get_value(param["sigma"]) * 0.1
        from scipy.optimize import fmin_l_bfgs_b
        sigma_value, sigma_likelihood_min, info = fmin_l_bfgs_b(sigma_loglikelihood, [K.get_value(param["sigma"])], approx_grad = True, bounds = [(lower, None)])
        K.set_value(param["sigma"], sigma_value[0])
    callbacks = [
        EpochCoordinateDescent(param, X, wp_bp, param_updater, input_output_preprocessor = lambda x : x, batch_size = batch_size)
    ]
    def mse_pred(x):
        sigma = K.get_value(param["sigma"])
        return np.exp(x + sigma * sigma / 2)
    def mae_pred(x):
        return np.exp(x)
    return (param, training_loss, loglikelihood, callbacks, mse_pred, mae_pred, np.nanmean(np.log(wp)))
