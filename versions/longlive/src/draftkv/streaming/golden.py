"""Independent scalar CPU dense and online-softmax references for static tests."""
import math


def dense_attention(q, k, v):
    scale = len(q[0]) ** -.5
    out = []
    for query in q:
        scores = [sum(a*b for a,b in zip(query,key))*scale for key in k]
        weights = [math.exp(s-max(scores)) for s in scores]
        out.append([sum(w*value[d] for w,value in zip(weights,v))/sum(weights) for d in range(len(v[0]))])
    return out


def online_attention(q, k, v, chunk_size):
    result = []
    for query in q:
        maximum,denominator,numerator = -math.inf,0.,[0.]*len(v[0])
        for start in range(0,len(k),chunk_size):
            scores = [sum(a*b for a,b in zip(query,key))*len(query)**-.5 for key in k[start:start+chunk_size]]
            newmax = max(maximum,max(scores)); correction = math.exp(maximum-newmax)
            weights = [math.exp(s-newmax) for s in scores]
            numerator = [old*correction+sum(w*value[d] for w,value in zip(weights,v[start:start+chunk_size])) for d,old in enumerate(numerator)]
            denominator = denominator*correction+sum(weights); maximum = newmax
        result.append([value/denominator for value in numerator])
    return result
