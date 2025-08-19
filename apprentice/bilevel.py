# import apprentice
import numpy as np

def normalize_weights(w):
    for weight in w:
        if weight < 0:
            raise ValueError("Weights must be non-negative")
    total = np.sum(w)
    return w / total

class InnerResult:
    def __init__(self, weights, parameter, g_value, inner_fun, err_per_obs, score_per_obs=None):
        self.weights = weights
        self.parameter = parameter
        self.g_value = g_value
        self.inner_fun = inner_fun
        self.err_per_obs = err_per_obs
        self.score_per_obs = score_per_obs

class BilevelObjective:
    def __init__(self, tuning_obj, objective="portfolio", lambda_var=1.0):
        self.tuning_obj = tuning_obj
        self.objective = objective
        self.lambda_var = lambda_var
        self.n_obs = len(self.tuning_obj._hnames)

    def setWeights(self, w):
        w = normalize_weights(w)
        wdict = {hn: wi for hn, wi in zip(self.tuning_obj._hnames, w)}
        self.tuning_obj.setWeights(wdict)
        return w
    
    def innerSolve(self, w, survey, restart):
        w = self.setWeights(w)
        res = self.tuning_obj.minimize(survey=1, restart=1, method="tnc", tol=1e-6, saddlePointCheck=False) 
        parameter_opt = res["x"] if isinstance(res, dict) else res.x
        return w, np.array(parameter_opt), res
    
    def modelValues(self, parameter, sel=None):
        x_full = self.to.mkPoint(parameter)
        return self.to._AS.vals(x_full, sel=sel)
    
    def perObservableBins(self, hname):
        return [i for i, b in enumerate(self.tuning_obj._binids) if b.startswith(hname)]
    
    def portfolioStats(self, parameter):
        err_per_obs = {}
        Y = self.tuning_obj._Y; E = self.tuning_obj._E
        for hn in self.tuning_obj._hnames:
            sel = self.perObservableBins(hn)
            fvals = self.modelValues(parameter, sel=sel)
            if self.tuning_obj._EAS is not None:
                x_full = self.tuning_obj.mkPoint(parameter)
                df = self.tuning_obj._EAS.vals(x_full, sel=sel)
            else:
                df = np.zeros_like(fvals)
            denominator = df * df + E[sel] * E[sel]
            mask = denominator > 0
            err_per_obs[hn] = float(np.mean((fvals[mask] - Y[sel][mask]) ** 2 / denominator[mask]))
        errs = np.array(list(err_per_obs.values()))
        mu = float(np.mean(errs))
        var = float(np.mean((errs - mu) ** 2))
        return err_per_obs, mu, var

    def scoreStats(self, parameter, mode):
        Y = self.tuning_obj._Y; E = self.tuning_obj._E
        out = {}
        for hn in self.tuning_obj._hnames:
            sel = self.perObservableBins(hn)
            fvals = self.modelValues(parameter, sel=sel)
            mask = E[sel] * E[sel] > 0
            sb = ((fvals - Y[sel]) ** 2) / (E[sel] * E[sel]) + np.log(E[sel] * E[sel])
            out[hn] = float(np.mean(sb[mask]) if mode == 'mean' else np.median(sb[mask]))
        return out
    
    def outerObjective(self, parameter):
        if self.objective == 'portfolio':
            err_per_obs, mu, var = self.portfolioStats(parameter)
            return mu + self.lambda_var * var, err_per_obs, None
        elif self.objective in ('score-mean', 'score-median'):
            mode = 'mean' if self.objective == 'score-mean' else 'median'
            score_per_obs = self.scoreStats(parameter, mode)
            g = float(sum(score_per_obs.values()))
            err_per_obs, _, _ = self.portfolioStats(parameter)
            return g, err_per_obs, score_per_obs
        else:
            raise ValueError(f"Unknown objective {self.objective}")
        
    def evaluate(self, w, survey, restart):
        w, parameter, res = self.innerSolve(w, survey, restart)
        g, err_per_obs, score_per_obs = self.outerObjective(parameter)
        inner_fun = float(res["fun"]) if isinstance(res, dict) else float(res.fun)
        IR = InnerResult(weights=w, parameter=parameter, g_value=g, inner_fun=inner_fun, err_per_obs=err_per_obs, score_per_obs=score_per_obs)
        self.history.append(IR)
        return IR

    def optimise(self):

    def bestResult(self):

    def toJson(self):


if __name__ == "__main__":
    import appset
    import bilevel

    WFILE = "Test-app-bilevel/weights.txt"
    DATA  = "Test-app-bilevel/data.json"
    APP   = "Test-app-bilevel/app_5_0.json"

    GOF = appset.TuningObjective2(WFILE, DATA, APP)
    BO  = bilevel.BilevelObjective(GOF)