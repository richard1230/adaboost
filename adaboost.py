from numpy import *

def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels



datMat, classLabels = loadSimpData()

#通过阈值比较对数据进行分类,参数分别为(数据集，dimen为第几列,threshVal为阈值,threshIneq为阈值变量(lt或gt,大于或者小于的一个符号))
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))                      #返回类别，shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数。
    # print(retArray)
    # [[1.]
    #  [1.]
    #  [1.]
    #  [1.]
    #  [1.]]
    #注：ones([5,1])是等价于ones((5,1))的，ones((shape(dataMatrix)[0], 1)) 等价于ones((5,1))
    # print(shape(dataMatrix))                                        #(5, 2)
    # print(shape(dataMatrix)[0])                                     #5

    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        #这里是把matrix里面的第一列即:[1,2,1.3,1,2]中的每个元素与[1.2,1.2,1.2,1.2,12]里面的对应的每个元素比较大小(这里假设threshVal值为1.2)，如果前者小于等于后者,则将[[1.],[1.],[1.],[1.],[1.]]中的对应的那个1改成-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
# #[[-1.]
#  [-1.]
#  [-1.]
#  [-1.]
#  [-1.]]retArray是类似于左边这样的
#dataMatrix[:,j]的用法
# d = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]])
# # print(d)
# # [[1 2 3 4]
# #  [2 3 4 5]
# #  [5 6 7 8]]
# d0 = d[:, 0]
# print(d0)
# # [1 2 5]  ;这是第0列
# e = d[:, 1]
# print(e)
# # [2 3 6]  ；这是第1列
#
# print("--------下面是f--------")
# f = d[:, 2]
# print(f)
# # [3 4 7]   ；这是第2列
# print("-------------")
# g = d[:, 3]
# print(g)
# [4 5 8]   ；这是第3列
# 总结:d[:,j],j是列的索引



#创建最佳决策树 参数分别为(数据集,类别,权值)，这里的dataArr，classLabels分别就是上面加载数据函数返回的datMat，classLabels
#这个函数的作用是利用足够多的阈值不断的去测试，最后获取足够小的错误率，以及此时最小错误率所对应的原始数据的那一列，以及此时的阈值,还有预测值(它本质上就是一个分类器)
#
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)                                                   #感觉没啥变化，dataArr本来就是矩阵
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)  # 5,2
    numSteps = 10.0                                                               #步数
    bestStump = {}                                                                #最优决策树(弱分类器)
    bestClasEst = mat(zeros((m, 1)))                                              #这里m等于5,这里初始值是0([[0],[0],[0],[0],[0]]),最终是要变为[[-1],[1],[-1],[-1],[1]]
    minError = inf
    for i in range(n):                                                            #这里的n等于2，其实这里是遍历每一列;  这里是决策树取列
        rangeMin = dataMatrix[:, i].min()                                         #i为0时，取的是第一列的最小值;i为1时,取的是第二列的最小值;特征最小的值
        rangeMax = dataMatrix[:, i].max()                                         #i为0时，取的是第一列的最大值;i为1时,取的是第二列的最大值;特征最大的值
        stepSize = (rangeMax - rangeMin) / numSteps                               #这里求的是步长,每一列的最大值减去最小值除以步数
        for j in range(-1, int(numSteps) ):                                        #j一开始取-1,这里给了一个范围，通过范围来设定阈值，通过不同的阈值来获取不同的错误率,
                                                                                  #>>> range(1, 11)【这里默认步长为1】     # 从 1 开始到 11；[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)                      #这里是利用一个简单的算法来确定阈值
                # print(threshVal)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  #这里的dataMatrix是数据集,i表示是第几列,threshVal表示的阈值,inequal表示的是符号
                                                                                  #predictedVals表示的是获取到预测的类别,这里得到的是多个预测值，下面就可以计算错误率了
                errArr = mat(ones((m, 1)))                                        #这里m等于5,这里是初始化错误率，下面判断哪个错误率是最小的
                # print(errArr)
                errArr[predictedVals == labelMat] = 0                             #当这里的预测值predictedVals等于真实值labelMat的时候,将errAr里面的对应值设为0(即判断对了的设置为0);这实际上是两个列表里面对应值的比较
                weightedError = D.T * errArr                                      #这个D.T是权值，这里其实计算的是错误率，
                # D.T * errArr = [[0.2] [0.2] [0.2] [0.2][0.2]] * [[0.] [0.] [1.][1.][0.]]
                # print(weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    # print(bestClasEst)
                    # print("-------")
                    bestStump['dim'] = i                                           #这里是决策树了，得出最优的那个列的索引
                    bestStump['thresh'] = threshVal                                #得到最优的那个阈值
                    bestStump['ineq'] = inequal                                    #得到最优的那个符号，stumpClassify函数的最后一个参数,这里符号有两个，一个是Lt,一个是gt
    return bestStump, minError, bestClasEst                                        #这里返回的是最优的决策树,最优的误差(误差值最小)，最优的预测值(分类器)【其本质上就是说分成这种分类的时候，其误差是最小的】
# ({'dim': 0, 'thresh': 1.3, 'ineq': 'lt'}, matrix([[0.2]]), array([[-1.],
#        [ 1.],
#        [-1.],
#        [-1.],
#        [ 1.]]))



D = mat(ones((5,1))/5)
# print(D)
# [[0.2]
#  [0.2]
#  [0.2]
#  [0.2]
#  [0.2]]
bestStump, minError, bestClasEst = buildStump(datMat, classLabels, D)
# print(buildStump(datMat,classLabels,D))


##adaboost决策树训练(迭代次数40)
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []                                       #这里面可以放多个分类器
    m = shape(dataArr)[0]
    # print(m)                                              #m 是 5
    # print("-----上面这是m-----")
    D = mat(ones((m, 1)) / m)                               #这里是在初始化权值
    aggClassEst = mat(zeros((m, 1)))                        #这里是话语权和分类器的乘积(这里是初始化为0了!),最终的分类器等于每个话语权与每个分类器的乘积之和
    for i in range(numIt):                                  #这里numIt等于40
        #获取分类器(最优决策树)，错误率，预测类别
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))                     #计算话语权,alpha = 1/2 *log)((1-error)/max(error,1e-16)),这里分母为max(error,1e-16)这么写的目的是不希望分母为0

        bestStump['alpha'] = alpha                                                      #把话语权添加到词典里面
        weakClassArr.append(bestStump)                                                  #词典添加到列表里面
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)                     #这里计算的是指数:expon=exp(-alpha_m * y_i * G_m(x_i));classLabels对应于公式里面的y_i(_符号表示i为下标)
                                                                                        #classEst表示的是分类的结果,是通过buildStump函数来或得的，对应于G_m(x_i)
        D = multiply(D, exp(expon))                                                     #这里是更新权值,W_m+1_i = W_mi/Z_m * exp(expon);W_m+1_i对应于左边的D;W_mi/Z_m对应于右边的D
        D = D / D.sum()                                                                 #这里是归一化操作;到这里更新完
        aggClassEst += alpha * classEst                                                 #更新完之后需要做的是构建分类器的线性组合,即f(x) = alpha_m * G_m(x)(注意:m取值为从1到)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))     #print(multiply(False,2))            #0
                                                                                        #print(multiply(True,2))             #2
                                                                                        #这里sign(aggClassEst) != mat(classLabels).T这个条件成立的话，就为1，否则为0，其实这里是判断预测错误(正确)的个数
        errorRate = aggErrors.sum() / m                                                 #错误的个数/总数 就会得到错误率，这里循环到第三次的时候errorRate等于0了
        if errorRate == 0.0: break
    return weakClassArr                                                                 #最终的分类器
    #[{'dim': 0, 'thresh': 1.3, 'ineq': 'lt', 'alpha': 0.6931471805599453}, {'dim': 1, 'thresh': 1.0, 'ineq': 'lt', 'alpha': 0.9729550745276565}, {'dim': 0, 'thresh': 0.9, 'ineq': 'lt', 'alpha': 0.8958797346140273}]


weakClassArr = adaBoostTrainDS(datMat, classLabels, numIt=40)
# print(weakClassArr)



#测试分类器
def adaClassify(datToClass, classifierArr):                                                 #datToClass是测试集,classifierArr是分类器的集合
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]                                                                #得到对应的形状
    aggClassEst = mat(zeros((m, 1)))                                                        #初始化话语权和分类器的乘积
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])                                  #这里得到的是第i个分类器
        aggClassEst += classifierArr[i]['alpha'] * classEst                                 #这里是第i个分类器所对应的alpha乘以所对应的分类器,而后相乘得到最终的分类器
    return sign(aggClassEst)

print(adaClassify([1., 2.1], weakClassArr))         #1
print(adaClassify([1.3, 1.], weakClassArr))         #-1



# print(datMat,classLabels)
# stumpClassify(datMat, 0, 1.2, 'lt')
# [[1.]
#  [1.]
#  [1.]
#  [1.]
#  # [1.]]