import matplotlib.pyplot as plt
import preprocess as prep

data = prep.dist_named

data.groupby(['GENDER'])['GENDER'].count().plot(kind="bar")
plt.show()


train_cred = prep.y_train
test_cred = prep.y_test

#plot both histograms(range from -10 to 10), bins set to 100
plt.hist([train_cred, test_cred], bins= 100, range=[400,900], alpha=0.5, label=['Train', 'Test'])
#plot legend
plt.legend(loc='upper right')
#show it
plt.show()

data = prep.data
data.groupby(["CRED"])["CRED"].count().plot(kind="bar")
plt.show()
