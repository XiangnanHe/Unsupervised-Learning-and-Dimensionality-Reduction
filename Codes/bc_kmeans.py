from sklearn import datasets

from clustertesters import KMeansTestCluster as kmtc

if __name__ == "__main__":
    breast_cancer = datasets.load_breast_cancer()
    #print breast_cancer
    X, y = breast_cancer.data, breast_cancer.target
    #print X


    tester = kmtc.KMeansTestCluster(X, y, clusters=range(1,30), plot=True, targetcluster=2, stats=True)
    tester.run()






