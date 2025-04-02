import numpy as np
import scipy as sc
import random
from scipy import linalg as la
from numpy.linalg import norm
import scipy.sparse as sparse
from scipy.sparse import rand as rndma
import time
import networkx as nx


class MAS_Approx:
    def __init__(self, prec=8, dim=50, theta=1e-4):
        self.precesion = prec #setting the rounding parameter
        self.Eps = 10**(-prec)
        self.dimension = dim #setting the dimension
        self.theta = theta #set the tolerance parameter
        # run_time = []
        # edge_perc = []
        # no_of_iter = []
        # for x in range(5):
        #     A = self.create_graph(self.dimension)
        #     # print(f"A: {A}")
        #     # print (f"leading(A): {leading(A)}")#generating start. graph
        #     GPN = self.closest_graph_PN(A) #running the algorithm 
        #     # print (f"GPN: {GPN}")
        #     run_time.append(GPN[0])
        #     edge_perc.append(GPN[1])
        #     no_of_iter.append(GPN[2])
        #     adj_matrix = GPN[-1]
        #     G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
        #     is_dag = nx.is_directed_acyclic_graph(G)
        #     # print(f"adj_matrix {adj_matrix}")
        #     print(f"is_dag {is_dag}")
        #     print ('......................')

    def run(self, Adj_matrix):
        GPN = self.closest_graph_PN(Adj_matrix) #running the algorithm 
        run_time = GPN[0]
        edge_perc = GPN[1]
        no_of_iter = GPN[2]
        DAG_adj_matrix = GPN[-1]
        G = nx.from_numpy_array(DAG_adj_matrix, create_using=nx.DiGraph())
        is_dag = nx.is_directed_acyclic_graph(G)
        assert is_dag
        # print(GPN)
        return run_time, edge_perc, no_of_iter, G

    def lp_solution(self, A,v,supp,tau):
        
        D = len(supp)
        X = np.copy(A)
        
        ind = np.argsort(v)[::-1]
        ind = ind[:D]
        
        for i in range(self.dimension):
            S = 0
            for l in ind:
                S += A[i,l]
                if (S <= tau):
                    X[i,l] = 0
                else:
                    X[i,l] = -tau + S
                    break
        
        return np.round(X,self.precesion)

    #Block 3: selective power method 
    def pwrmthd(self, A):
        A = A + np.identity(self.dimension) #see the remark on page 28
        v0 = np.array([1 for i in range(self.dimension)]) #starting vector of all ones 
        v1 = np.dot(A,v0)/float(norm(np.dot(A,v0)))
        v1 = np.round(v1,self.precesion)
        while norm(v0-v1) > self.Eps*10: #the precision parametar $\varepsilon$ 
            v0 = v1
            v1 = np.dot(A,v0)/float(norm(np.dot(A,v0)))
        return np.round(v1,self.precesion-1)

    #Block 7: implementing the the greedy method for min. on the ball of radius k (Step 1)
    def selective_greedy(self, A,tau):
        
        X = np.copy(A)
        v0 = self.pwrmthd(X) #computing the leading eigenvalue
        supp = list(np.where(v0 != 0)[0]) #getting the support
        notsupp = list(set(range(self.dimension)) - set(supp))
        notsupp.sort()

        while True: #constructing the solution X_k
            Z = np.copy(X)
            v = v0
            X = self.lp_solution(A,v,supp,tau)
            X[notsupp] = Z[notsupp]
            
            for k in supp:
                olddot = np.dot(Z[k],v)
                newdot = np.dot(X[k],v)
                if (olddot < newdot) or (abs(olddot - newdot) < 1e-7): #see page X
                    X[k] = Z[k]
            
            v0 = self.pwrmthd(X)
            spect_radius = np.round(self.leading(X),self.precesion)
            
            '''if matrices of iterations k-1 and k match on the support, 
            OR if they have the same leading eigenvector,
            OR if the spectral radius of X_k is less than 1, we finish the greedy method''' 
            if (X[supp] == Z[supp]).all() or (v == v0).all() or (spect_radius < 1):
                return np.round(X,self.precesion), spect_radius
            else:
                supp = list(np.where(v0 != 0)[0])
                notsupp = list(set(range(self.dimension)) - set(supp))
                notsupp.sort()

    #Block 8: implementing the the greedy method for minimization on the ball 
    #of radius \tau (Step 1)
    def selective_greedylinf(self, A,tau):
        
        X = np.copy(A)
        v0 = self.pwrmthd(X) #computing the leading eigenvalue
        supp = list(np.where(v0 != 0)[0]) #getting the support
        notsupp = list(set(range(self.dimension)) - set(supp))
        notsupp.sort()

        while True: #constructing the solution X_k
            Z = np.copy(X)
            v = v0
            X = self.lp_solution(A,v,supp,tau)
            X[notsupp] = Z[notsupp]
            
            for k in supp:
                olddot = np.dot(Z[k],v)
                newdot = np.dot(X[k],v)
                if (olddot < newdot) or (abs(olddot - newdot) < 1e-6): #see the Appendix
                    X[k] = Z[k]
            
            v0 = self.pwrmthd(X)
            spect_radius = self.leading(X)
            
            '''if matrices of iterations k-1 and k match on the support, 
            OR if they have the same leading eigenvector,
            OR if the spectral radius of X_k is less than 1 - \theta, 
            we finish the greedy method'''
            if (X[supp] == Z[supp]).all() or (v == v0).all() or (spect_radius < 1 - self.theta):
                supp = list(np.where(v0 != 0)[0])
                return np.round(X,self.precesion), spect_radius, v0, supp
            else:
                supp = list(np.where(v0 != 0)[0])
                notsupp = list(set(range(self.dimension)) - set(supp))
                notsupp.sort()

    def forward(self, A,spectfin,k):
        
        while (spectfin != 0):
            
            if (spectfin > 6): 
                k += 2
            else:
                k += 1
            
            Xstar, spectfin = self.selective_greedy(A,k)
            
        
        return Xstar, spectfin

    def backward(self, A,X,spect,k):
        
        while (spect == 0):
            Xstar = np.copy(X)
            spectfin = spect
            k -= 1
            X, spect = self.selectivegreedylinf(A,k)
            
        
        return Xstar, spectfin

    def the_tree(self, A,X):
        
        
        tree = []
        vertices = [i for i in range(self.dimension)]
        ind = [i for i in range(self.dimension)]
        Aredux_col = np.copy(A)
        Xredux_row = np.copy(X)
        Xredux_col = np.copy(X)
        
        while (vertices != []):
        
            #finding the source(s)
            sources = [j for j in vertices if (np.sum(Xredux_row[:,j]) == 0)]
            indy = np.argmax([np.sum((Aredux_col - Xredux_col)[j]) for j in sources])
            
            the_source = sources[indy]
            
            #taking theem out
            tree.append(the_source)
            vertices.remove(the_source)
            Aredux_col = A[np.ix_(ind,vertices)]
            Xredux_col = A[np.ix_(ind,vertices)]
            Xredux_row = X[np.ix_(vertices,ind)]
        
        for x in range(len(tree)):
            i = tree.pop(0)
            for j in tree:
                X[i,j] = A[i,j]

        return X

    def ubrmthd(self, A):
        evals, evecs = np.linalg.eig(A)
        evals = np.real(evals)
        evals = np.round(evals,self.precesion)
        evecs = np.round(evecs,self.precesion)
        rho = np.amax(evals)
        rho = np.round(rho,self.precesion)
        return evals, rho, evecs

    def perm(self, ind, D):
        
        P = np.zeros((D,D))
        
        for i in range(D):
            P[i,ind[i]] = 1
        
        return P

    def minimal(self, A):
        evals, rho, evecs = self.ubrmthd(A)
        index = np.where(evals == rho)[0]
        
        dic = {} #finding the leading eigenvectors and counting their support
        nonzeros = []
        for i in index:
            eigenvec = evecs[:,i]
            l = len(np.where(eigenvec != 0)[0])
            nonzeros.append(l)
            dic[l] = i
            
        k = dic[np.amin(nonzeros)] #fining the minimal eigenvector
        v = np.real(evecs[:,k])
        v = np.abs(v)
        v = np.round(v,self.precesion)
        supp = list(np.where(v != 0)[0])
        return v, supp

    def transfer(self, l,m,x): #transfers element x from list l to the list m
        l.remove(x)
        m.append(x)
        
        return l, m    

    def isinbasic(self, A,index):
        
        
        i = index[0] #index that is allegedly in basic set
        void = np.where(A.T[:,i] == 0)[0] #potential void
        void = list(void)
        if i in void:
            void.remove(i)
        
        bset = [j for j in index if j not in void] #potential basic set
    
        
        
        
        bset1 = bset #set to test verices on
        while True:
            bset2 = [] #set for new vertices
            void0 = list(void)
            for k in void0:
                check = [A.T[k,j] == 0 for j in bset1]
                if not all(check):
                    void, bset2 = self.transfer(void,bset2,k)
            if (void0 == void):
                break
            else:
                bset += bset2
                bset1 = bset2 #now testing on new vertices


        bset.sort()
        void.sort()           

        return bset, void

    def irreduce(self, A,bset):

        D = len(bset)
        C = A[np.ix_(bset,bset)]
        if (len(C) != 0):
            spectC = self.ubrmthd(C)[1]
        else:
            return True
        spectA = self.ubrmthd(A)[1]
        if (spectC == spectA):
            K = np.linalg.matrix_power(np.identity(D) + C, D-1)
            return (K > 0).all()
        else:
            return False  

    def isinvoid(self, A,index):
        
        i = index[0]
        
        
        void = np.where(A.T[i] != 0)[0] #potential void
        void = list(void)
        if i not in void:
            void.append(i)
            
        bset = [j for j in index if j not in void]
        
        void0 = list(set(void) - set([i])) #set to test verices on
        while True:
            void1 = [] #a void to be
            for k in void0:
                newvoid = [j for j in bset if A.T[k,j] != 0]
                for x in newvoid:
                    bset, void = self.transfer(bset,void,x)
                void1 += newvoid
            if (void1 == []) or (bset == []):
                break
            else:
                void0 = void1

        bset.sort()
        void.sort()
        
        return bset, void

    def basicset(self, A,D):
        
        index = range(D) 
        
        spect = self.ubrmthd(A)[1] #checking for loops 
        loops = [A[i,i] for i in range(D)]
        frutiloop = list(np.where(spect == loops)[0])
        if (frutiloop != []):
            S = False
            bset = frutiloop
            void = list(set(index) - set(frutiloop))
        else:
            S = True
            void = []
            
        while S:
            bset, void0 = self.isinbasic(A,index) #potential basic and void
            if self.irreduce(A,bset):
                void += void0
                void = list(set(void))
                break
            else:
                bset, void0 = self.isinvoid(A,index)
                bset.sort()
                index = bset
                void += void0
                void = list(set(void))
                if self.irreduce(A,bset):
                    break
                    
        
                    
                
        bset.sort()
        void.sort()
        ind = void + bset
        
        P = self.perm(ind,len(ind))
        Kac = np.matmul(P,np.matmul(A,P.T))
    
        return bset #getting the basic set

    def PN_greedy(self, A,tau):
        
        start = time.time()
        global it

        X = np.copy(A)
        v0, supp0 = self.minimal(X) #computing the minimal eigenvec and its support
        it += 1
        
        while True:
            
            Z = np.copy(X)
            supp = supp0
            v = v0
            notsupp = list(set(range(self.dimension)) - set(supp))
            notsupp.sort()
            X = self.lp_solution(A,v,supp,tau)
            X[notsupp] = Z[notsupp]
            
            optin = supp[:]
            
            for k in supp:
                olddot = np.dot(Z[k],v)
                newdot = np.dot(X[k],v)
                if (olddot < newdot) or (abs(olddot - newdot) < 1e-5): #see the Appendix
                    X[k] = Z[k]
                    optin.remove(k)
        
            
            v0, supp0 = self.minimal(X)
            it += 1
            if (set(supp0) == set(optin)) or (v == v0).all():
                spectra = self.leading(X)
                run = time.time() - start
                # print (run, it)
                return X, spectra
            
            else: #comparing basic set with set of optimal indices
                
                Xredux = X[np.ix_(supp,supp)]
                bsetredux = self.basicset(Xredux,len(supp)) #basic set
                bset = [supp[k] for k in bsetredux]
                if (set(bset) <= set(optin)):
                    v0 = np.zeros(self.dimension)
                    vredux, suppredux = self.minimal(Xredux)
                    v0[supp] = vredux #the real eigenvec
                    supp0 = [supp[k] for k in suppredux]
                else:
                    v0, supp0 = self.minimal(X)

    #Block 5: function for computing the spectral radius
    def leading(self, A):
        
        evals = np.linalg.eig(A)[0] #set of eigenvalues 
        return np.amax(np.real(evals)) #spectral radius

    def closest_graph_PN(self, A):
        

        start = time.time()
        global it
        it = 0
        

        row_sums = [np.sum(A[i]) for i in range(self.dimension)]  

        k0 = np.amax(row_sums)
        # print(k0)
        k0 = np.trunc(k0/2)
        k1 = np.amin(row_sums)
        seg = k0
        
        '''doing a bisection in k until we obtain a matrix with
        appropriate spectral radius'''

        while (seg >= 1):

            Xstar, spect_radius = self.PN_greedy(A,k0)
            # print(spect_radius, k0)
            

            
            if (spect_radius > 3):
                k1 = k0
                seg /= 2.0
                seg = np.ceil(seg)
                k0 += seg

            elif (spect_radius == 0):
                k1 = k0
                seg /= 2.0
                seg = np.ceil(seg)
                k0 -= seg

            else:
                k1 = k0
                break

        if (spect_radius != 0):
            '''if a last obtained matrix has a spectral radius bigger than zero
            we move k forward untill we obtain an acyclic graph'''
            Xstar, spect_radius = self.forward(A,spect_radius,k1)
        else:
            '''if a last obtained matrix has a zero spectral radius, 
            we move k backwards untill we get a minimal k for which
            we have an acyclic graph'''
            Xstar, spect_radius = self.backward(A,Xstar,spect_radius,k1)


        run = time.time() - start #running time after the Step X
        run = np.round(run,2)
        
        #computing the percentage of saved edges after the Step X
        perc = (np.sum(Xstar)/np.sum(A)) 
        
        Z = np.copy(Xstar)
        
        '''unning the DFS algorithm and restoring the edges;
        this will give us the MAS approximation'''
        Gamma = self.the_tree(A,Z) 
        # print(f"Gamma {Gamma}")


        run_tree = time.time() - start #running time after the Step X
        run_tree = np.round(run_tree,2)
        
        '''computing the percentage of saved edges;
        this will actuall give us how good our MAS approximation is'''
        perc_tree = (np.sum(Gamma)/np.sum(A))
        
        return run_tree, perc_tree, it, norm(Gamma - A,np.Inf), Gamma

    def create_graph(self, dim):
        
        G = nx.watts_strogatz_graph(dim,25,0.1)
        
        # print(G)
        # print(type(G))
        G = nx.adjacency_matrix(G).toarray()
        # print(G)
        
        
        #making the graph directed
        
        for i in range(dim):
            for j in range(i):
                if G[i,j] == 1:
                    cut = random.choice(['a','b'])
                    if cut == 'a':
                        G[i,j] = 0
                    elif cut == 'b':
                        G[j,i] = 0

        return G



if __name__ == '__main__':
    mas = MAS_Approx(prec=8, dim=50, theta=1e-4)

    for x in range(1):
        A = mas.create_graph(mas.dimension)
        print(A, A.shape)
        run_time, edge_perc, no_of_iter, G = mas.run(A)
        print(f"{run_time} {edge_perc} {no_of_iter}")
        print(f"DAG\n{G}")
    
# print (np.average(run_time), np.average(edge_perc), np.average(no_of_iter))