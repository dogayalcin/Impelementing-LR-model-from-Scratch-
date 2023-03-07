class LineerRegression:
    learning_rate = []
    epoch = None
    loss_list = None
    accuracy_list = None
    z_list = []
    m1 = None
    m2 = None
    b = None
    
    def __init__(self, learning_rate: float, epoch: int):
        self.learning_rate = learning_rate
        self.epoch = epoch

    def fit(self, x_train, y_train, z_train):
        #baslangic deger atamalari
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        self.loss_list = []
        self.accuracy_list = []
        for e in range(0,self.epoch):
            turev_m1 = 0
            turev_m2 = 0
            turev_b = 0
            m1 = self.m1
            m2 = self.m2
            b = self.b
            learning_rate = self.learning_rate
            #m1, m2 ve b degerlerinin Loss fonksiyonuna gore turevi hesaplanir
            for i in range(0, len(x_train)):
                turev_m1 += (m1*x_train[i] + m2*y_train[i] + b - z_train[i])*x_train[i]
                turev_m2 += (m1*x_train[i] + m2*y_train[i] + b - z_train[i])*y_train[i]
                turev_b += m1*x_train[i] + m2*y_train[i] + b - z_train[i]

            turev_m1 = (2/len(x_train))*turev_m1
            turev_m2 = (2/len(x_train))*turev_m2
            turev_b = (2/len(x_train))*turev_b
            #Agirliklar Loss sonucunu minimize edecek sekilde guncellenir
            self.m1 = m1 - learning_rate*turev_m1
            self.m2 = m2 - learning_rate*turev_m2
            self.b = b - learning_rate*turev_b

            self.z(x_train, y_train)
            #loss
            self.loss_function(z_train)
            #accuracy
            self.rsquare(z_train, self.z_list)

    def predict(self, x_test, y_test):
        self.z(x_test, y_test)
        return self.z_list
        
    def rsquare(self, z, z_predicted):
        mean = 0
        for i in range (0, len(z)):
            mean += z[i]
    
        mean = mean/len(z)
        rsquare = 0
        m = 0
        n = 0
        for i in range (0, len(z)):
             m += (z[i] - z_predicted[i])*(z[i] - z_predicted[i])
             n += (z[i] - mean)*(z[i] - mean)
        
        rsquare = 1 - (m/n)
        self.accuracy_list.append(rsquare)
        return rsquare
    
    def z(self, x, y):
        z_list = []
        for i in range(0, len(x)):
            z = self.m1*x[i] + self.m2*y[i] + self.b
            z_list.append(z)
        self.z_list = z_list
    
    def loss_function(self, z):
        loss = 0
        for i in range(0, len(z)):
            loss += (z[i] - self.z_list[i])*(z[i] - self.z_list[i])
        loss = loss/len(z)
        self.loss_list.append(loss)
        
    def getLoss(self):
        return self.loss_list
    
    def getAccuracy(self):
        return self.accuracy_list