import numpy as np
import json
try:
    from scipy.interpolate import Rbf as SciPyRbf
except Exception:
    SciPyRbf = None

def normalize_weights(w):
    for weight in w:
        if weight < 0:
            raise ValueError("Weights must be non-negative")
    total = np.sum(w)
    return w / total

class InnerResult:
    def __init__(self, weights, parameter, g_value, inner_fun, res, err_per_obs, score_per_obs=None):
        self.weights = weights
        self.parameter = parameter
        self.g_value = g_value
        self.inner_fun = inner_fun
        self.res = res
        self.err_per_obs = err_per_obs
        self.score_per_obs = score_per_obs

class BilevelObjective:
    def __init__(self, tuning_obj, objective="portfolio", lambda_var=1.0):
        self.tuning_obj = tuning_obj
        self.objective = objective
        self.lambda_var = lambda_var
        self.n_obs = len(np.unique(self.tuning_obj._hnames))
        self.obs_names = np.unique(self.tuning_obj._hnames)
        self.history = []
        try:
            self.rng = np.random.default_rng()
        except Exception:
            self.rng = np.random.RandomState()

    def setWeights(self, w):
        w = normalize_weights(w)
        wdict = {hn: wi for hn, wi in zip(self.obs_names, w)}
        self.tuning_obj.setWeights(wdict)
        return w
    
    def innerSolve(self, w, survey, restart):
        w = self.setWeights(w)
        res = self.tuning_obj.minimize(survey, restart, method="tnc", tol=1e-6, saddlePointCheck=False)
        parameter_opt = res["x"] if isinstance(res, dict) else res.x
        return w, np.array(parameter_opt), res
    
    def modelValues(self, parameter, sel=None):
        x_full = self.tuning_obj.mkPoint(parameter)
        return self.tuning_obj._AS.vals(x_full, sel=sel)
    
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
        IR = InnerResult(weights=w, parameter=parameter, g_value=g, inner_fun=inner_fun, res=res, err_per_obs=err_per_obs, score_per_obs=score_per_obs)
        self.history.append(IR)
        return IR

    def optimize(self, n_0=1, n_max=20, survey_inner=1, restart_inner=1, surrogate=True, candidates=1000, exploration=(0.2, 0.5, 0.8), debug=False):
        d = self.n_obs
        if candidates is None or candidates <= 0:
            candidates = 1
            surrogate = False
            if debug: print("Warning, candidates <= 0; set to 1 and disabled surrogate")
        if candidates < d * 500:
            if debug: print(f"Warning, {candidates} = candidates < d * 500 = {d * 500}, consider increasing")
        if surrogate and d > 300:
            if debug: print(f"Warning, Disabling surrogate (n_obs={d} too large)")
            surrogate = False

        def reduce_dim(W):
            return W[:, : max(1, d - 1)] if d > 1 else W.copy()

        class CubicRbf:
            def __init__(self, X, y, nugget=1e-10):
                self.X = np.asarray(X, float)
                self.y = np.asarray(y, float)
                self.n, self.p = self.X.shape if self.X.ndim == 2 else (self.X.shape[0], 1)
                D = self._cdist(self.X, self.X)
                Phi = D ** 3
                P = np.hstack([self.X, np.ones((self.n, 1))])
                top = np.hstack([Phi + nugget * np.eye(self.n), P])
                bottom = np.hstack([P.T, np.zeros((self.p + 1, self.p + 1))])
                K = np.vstack([top, bottom])
                rhs = np.concatenate([self.y, np.zeros(self.p + 1)])
                sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
                self.gamma = sol[: self.n]
                self.beta = sol[self.n :]

            def _cdist(self, A, B):
                A2 = np.sum(A * A, axis=1, keepdims=True)
                B2 = np.sum(B * B, axis=1, keepdims=True).T
                D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)
                return np.sqrt(D2)

            def predict(self, Xq):
                Xq = np.atleast_2d(Xq)
                D = self._cdist(Xq, self.X)
                Phi = D ** 3
                Pq = np.hstack([Xq, np.ones((Xq.shape[0], 1))])
                return Phi @ self.gamma + Pq @ self.beta
            
        class SciPyRbfWrapper:
            def __init__(self, X, y):
                p = X.shape[1] if X.ndim == 2 else 1
                coords = [X[:, j] for j in range(p)]
                self.rbf = SciPyRbf(*coords, y, function='cubic', smooth=1e-12)
                self.p = p

            def predict(self, Xq):
                Xq = np.atleast_2d(Xq)
                coords = [Xq[:, j] for j in range(self.p)]
                return self.rbf(*coords)
            
        if n_0 < d + 1:
            if debug: print(f"Warning, n_0 < d + 1; setting n_0 = {d + 1}")
        n_0 = max(n_0, d + 1)
        for i in range(n_0):
            w_0 = self.rng.dirichlet(np.ones(d)) if d > 1 else np.array([1.0])
            IR = self.evaluate(w_0, survey=survey_inner, restart=restart_inner)
            if debug: print(f"[init {i + 1}/{n_0}] g={IR.g_value:.6g}")

        if isinstance(exploration, (float, int)):
            nu_seq = [float(exploration)]
        else:
            nu_seq = [float(x) for x in exploration]
            if len(nu_seq) == 0:
                nu_seq = [0.5]
        nu_seq = [min(1.0, max(0.0, x)) for x in nu_seq]

        def fit_rbf():
            if not surrogate:
                return None
            if len(self.history) < max(2, d):
                return None
            W = np.array([h.weights for h in self.history], dtype=float)
            G = np.array([h.g_value for h in self.history], dtype=float)
            X = reduce_dim(W)
            try:
                if SciPyRbf is not None:
                    return SciPyRbfWrapper(X, G)
                return CubicRbf(X, G)
            except Exception as e:
                if debug:
                    print(f"Warning, RBF fit failed ({e}), using random candidate")
                return None

        if n_max < n_0 + 1:
            print(f"Warning, n_max < n_0 + 1; setting n_max = {n_0 + 20}")
            n_max = n_0 + 20
        for n in range(n_0, n_max):
            rbf = fit_rbf()
            Candidates = self.rng.dirichlet(np.ones(d), size=candidates) if d > 1 else np.ones((candidates, 1))

            if rbf is None:
                w_next = Candidates[0]

            else:
                Xc = reduce_dim(Candidates)

                s_pred = rbf.predict(Xc)
                smin = float(np.min(s_pred))
                smax = float(np.max(s_pred))
                V_s = (s_pred - smin) / (smax - smin) if smax > smin else np.zeros_like(s_pred)

                S = np.array([h.weights for h in self.history], dtype=float)
                Xs = reduce_dim(S)
                diff = Xc[:, None, :] - Xs[None, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=2))
                dmin = np.min(dists, axis=1)
                dmn = float(np.min(dmin))
                dmx = float(np.max(dmin))
                V_d = (dmx - dmin) / (dmx - dmn) if dmx > dmn else np.ones_like(dmin)

                nu = nu_seq[(n - n_0) % len(nu_seq)]
                V = nu * V_s + (1.0 - nu) * V_d
                w_next = Candidates[int(np.argmin(V))]

            IR = self.evaluate(w_next, survey=survey_inner, restart=restart_inner)
            if debug:
                print(f"[iter {(n - n_0) + 1}/{(n_max - n_0)}] g={IR.g_value:.6g}")

        return min(self.history, key=lambda r: r.g_value)

    def writeResults(self, wfile, outdir):
        if not hasattr(self, "history") or len(self.history) == 0:
            return None
        best = min(self.history, key=lambda r: r.g_value)
        w_best = np.asarray(best.weights, dtype=float)
        maxw = float(np.max(w_best)) if w_best.size else 0.0
        scale = (1.0 / maxw) if maxw > 0 else 1.0
        weight_map = {hn: float(w) * scale for hn, w in zip(self.obs_names, w_best)}
        try:
            out_path = f"{outdir}/best_weights.txt"
            with open(wfile, "r") as fin, open(out_path, "w") as fout:
                for line in fin:
                    raw = line.rstrip("\n")
                    if not raw.strip() or raw.lstrip().startswith('#'):
                        fout.write(line)
                        continue
                    parts = raw.split(None, 2)
                    if len(parts) < 2:
                        fout.write(line)
                        continue
                    name = parts[0]
                    rest = parts[2] if len(parts) >= 3 else ""
                    if len(name.split('#')) > 1:
                        fout.write(line)
                        continue
                    if name in weight_map:
                        new_w = weight_map[name]
                        new_line = f"{name} {new_w:.3f}" + (f" {rest}" if rest else "")
                        fout.write(new_line + "\n")
                    else:
                        fout.write(line)
            with open(f"{outdir}/history.json", "w") as f:
                f.write(self.toJson())
        except Exception:
            pass
        return 

    def toJson(self):
        if not hasattr(self, "history"):
            return json.dumps({"history": []})
        def innerResult_to_dict(IR):
            return {
                "weights": IR.weights.tolist() if isinstance(IR.weights, np.ndarray) else list(IR.weights),
                "parameter": IR.parameter.tolist() if isinstance(IR.parameter, np.ndarray) else list(IR.parameter),
                "g_value": float(IR.g_value),
                "inner_fun": float(IR.inner_fun),
                "err_per_obs": IR.err_per_obs,
                "score_per_obs": IR.score_per_obs,
            }
        if self.objective == 'portfolio':
            data = {
            "objective": self.objective,
            "lambda_var": self.lambda_var,
            "n_obs": int(self.n_obs),
            "history": [innerResult_to_dict(h) for h in self.history],
            }
        else:
            data = {
            "objective": self.objective,
            "n_obs": int(self.n_obs),
            "history": [innerResult_to_dict(h) for h in self.history],
            }
        return json.dumps(data, indent=2)