"""
  ============================================================================
  :class:`dist` -- Distance metrics
  ============================================================================

  This module provides a framework to calculate several types of distance
  metrics to compare two (N,) arrays.

  .. Copyright 2016 Efrain Hernandez-Rivera
       Last updated: 2016-09-12 by E. Hernandez-Rivera
       

  Funding Acknowledgement:
  .. This research was supported in part by an appointment to the Postgraduate
  .. Research Participation Program at the U.S. Army Research Laboratory
  .. administered by the Oak Ridge Institute for Science and Education (ORISE)
  .. through an interagency agreement between the U.S. Department of Energy
  .. and USARL.
"""

import numpy as np
from math import sqrt,log

class Distances(object):
    """ Python distance/similarity module. Currently, includes distances
        from Cha [1].

        Paramaters
        ----------
        family: name of the distance family
          minkowski
          L1
          inter
          inner
          fidelity
          squaredL2
          shannon
          combination
          vicissitude

        X: array_like
             Reference histogram/distribution
        Y: array_like
             Histogram/distribution to measure distance from X

        Returns
        ----------
        distances: dict
             Dictionary containing distances for all the family members as
             outlined by Cha

        Usage
        ----------

        >>> import PyDIST as distance
        >>> dist=distance.Distances([1,2,3],[4,6,8])

        >>> mink=dist.minkowski()

        References
        ----------
        [1] Cha, S.H, IJMMMAS, v. 1, iss. 4, pp. 300-307 (2007)
        [2] Hernandez-Rivera, et al. ACS Comb Sci, accepted (2016)
    """

    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."

        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)

        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20

        self.P=P
        self.Q=Q

    def minkowski(self,n=1):
        P=self.P; Q=self.Q
        return {'Euclidean' :sqrt(sum((P-Q)*(P-Q))),\
                'City Block':sum(abs(P-Q)),\
                'Minkowski' :(sum(abs(P-Q)**n))**(1./n),\
                'Chebyshev' :max(abs(P-Q))}

    def L1(self):
        P=self.P; Q=self.Q; A=sum(abs(P-Q)); d=len(P)
        return {'Sorensen'  :A/sum(P+Q),\
                'Gower'     :A/d,\
                'Sorgel'    :A/sum(np.maximum(P,Q)),\
                'Kulczynski':A/sum(np.minimum(P,Q)),\
                'Canberra'  :sum(abs(P-Q)/(P+Q)),\
                'Lorentzian':sum(np.log(1+abs(P-Q)))}

    def intersection(self):
        P=self.P; Q=self.Q; A=sum(abs(P-Q)); maxPQ=sum(np.maximum(P,Q))
        return {'Intersection':0.5*A,\
                'Wave Hedges' :sum(abs(P-Q)/np.maximum(P,Q)),\
                'Czekanowski' :A/sum(P+Q),\
                'Motyka'      :maxPQ/sum(P+Q),\
                'Ruzicka'     :1-sum(np.minimum(P,Q))/maxPQ,\
                'Tanimoto'    :sum(np.maximum(P,Q)-np.minimum(P,Q))/maxPQ}

    def inner(self):
        P=self.P; Q=self.Q; ip=sum(P*Q); p2=sum(P*P); q2=sum(Q*Q); d=len(P)
        return {'Inner Product':1-ip,\
                'Harmonic Mean':1-2.*sum(P*Q/(P+Q)),\
                'Cosine'       :1-ip/(sqrt(p2)*sqrt(q2)),\
                'Jaccard'      :sum((P-Q)*(P-Q))/(p2+q2-ip),\
                'Dice'         :sum((P-Q)*(P-Q))/(p2+q2)}

    def fidelity(self):
        P=self.P; Q=self.Q; fid=sum(np.sqrt(P*Q))
        return {'Fidelity'     :1-fid,\
                'Bhattacharyya':-log(fid),\
                'Hellinger'    :2*sqrt(1-fid),\
                'Matusita'     :sqrt(2-2*fid),\
                'Squared-Chord':sum((np.sqrt(P)-np.sqrt(Q))**2)}

    def squaredL2(self):
        P=self.P; Q=self.Q; d=len(P)
        return {'Squared Euclidean':sum((P-Q)**2),\
                'Pearson Chi':sum((P-Q)**2/Q),\
                'Neyman Chi' :sum((P-Q)**2/P),\
                'Squared Chi':sum((P-Q)**2/(P+Q)),\
                'Prob Symm'  :2*sum((P-Q)**2/(P+Q)),\
                'Divergence' :2*sum((P-Q)**2/(P+Q)**2),\
                'Clark'      :sqrt(sum((abs(P-Q)/(P+Q))**2)),\
                'Additive Symm':sum((P-Q)**2*(P+Q)/(P*Q))}

    def shannon(self):
        P=self.P; Q=self.Q
        return {'Kull-Leiber':sum(P*np.log(P/Q)),\
                'Jeffreys'   :sum((P-Q)*np.log(P/Q)),\
                'Kdivergence':sum(P*np.log(2*P/(P+Q))),\
                'Topsoe'     :sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q))),\
                'Jensen-Shan':0.5*sum(P*np.log(2*P/(P+Q))\
                                     +Q*np.log(2*Q/(P+Q))),\
                'Jensen-Diff':0.5*sum(P*np.log(P)+Q*np.log(Q)\
                                      -(P+Q)*np.log((P+Q)/2.))}

    def combination(self):
        P=self.P; Q=self.Q
        return {'Taneja'    :0.5*sum((P+Q)*np.log((P+Q)/(2.*np.sqrt(P*Q)))),\
                'Kumar-John':sum((P*P-Q*Q)**2/(2*(P*Q)**(1.5))),\
                'AverageL'  :0.5*(sum(abs(P-Q))+max(abs(P-Q)))}

    def vicissitude(self):
        P=self.P; Q=self.Q; p=sum((P-Q)*(P-Q)/P); q=sum((P-Q)*(P-Q)/Q)
        pqmin=np.minimum(P,Q)
        return {'Vicis-Wave Hedge':sum(abs(P-Q)/pqmin),\
                'Vicis-Symm Chi1' :sum((P-Q)*(P-Q)/pqmin**2),\
                'Vicis-Symm Chi2' :sum((P-Q)*(P-Q)/pqmin),\
                'Vicis-Symm Chi3' :sum((P-Q)*(P-Q)/np.maximum(P,Q)),\
                'Max-Symm Chi'    :max(p,q),\
                'Min-Symm Chi'    :min(p,q)}
