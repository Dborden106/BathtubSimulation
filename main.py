import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import statistics


def estimate(sum, x0, constraints, bounds):
    print("Best Fit")
    cons = {'type': 'ineq', 'fun': constraints}
    bestFit = minimize(sum, x0, method='SLSQP', bounds=bounds, constraints=cons)
    print("LSfitRules =", bestFit.x)
    return bestFit


def prediction(dataSet, length, length90, fitEquation, figure1, figure2):
    print("Prediction")
    yFit = []
    for i in range(0, length):
        yFit.append(fitEquation(i))
    print("YFitList =", yFit)
    figure1.plot(x1, yFit, color='r')
    figure1.vlines(length90, 0.98, 1.08, linestyles="dashed", colors="k")
    figure2.plot(x1, dataSet, drawstyle='steps-post', color='b')
    figure2.plot(x1, yFit, color='r')
    figure2.vlines(length90, 0.98, 1.08, linestyles="dashed", colors="k")
    return yFit


def quality(dataSet, length, length90, yFit):
    print("Measure of the quality of the fit")
    SSE = 0
    for i in range(0, length):
        SSE += math.pow((dataSet[i] - yFit[i]), 2)
    print("SSE =", SSE)
    PSSE = 0
    for i in range(length90, length):
        PSSE += math.pow((dataSet[i] - yFit[i]), 2)
    print("PSSE =", PSSE)
    # Mean Squared Error
    MSE = SSE / length
    print("MSE =", MSE)
    # Variance
    s2 = SSE / (length - 2)
    print("S2 =", s2)
    # r2
    tempSum = 0
    tempMean = statistics.mean(dataSet)
    for i in range(0, length):
        tempSum += math.pow((dataSet[i] - tempMean), 2)
    r2 = (tempSum - SSE) / tempSum
    print("r2 =", r2)
    p = 3
    r2Adj = 1 - (1 - r2) * (length - 1) / (length - p - 1)
    print("r2Adj =", r2Adj)
    return s2


def confidence(dataSet, length, length90, figure3, figure4, figure5, yFit, s2):
    print("Confidence Intervals")
    # Define lower 0.25 and upper 97.5 confidence levels of normal distribution
    # 99% CI
    normal025 = norm.ppf(0.025)
    print("normal025 =", normal025)
    normal975 = norm.ppf(0.975)
    print("normal975 =", normal975)
    # Contruct a 95% confidence interval for the Y predicted based on the normal approximation
    YCIlower = []
    for i in range(0, length):
        YCIlower.append(yFit[i] + normal025 * math.sqrt(s2))
    print("YCIlower =", YCIlower)
    YCIupper = []
    for i in range(0, length):
        YCIupper.append(yFit[i] + normal975 * math.sqrt(s2))
    print("YCIupper =", YCIupper)
    figure3.plot(x1, YCIlower, color='b')
    figure3.vlines(length90, 0.98, 1.08, linestyles="dashed", colors="k")
    figure4.plot(x1, YCIupper, color='b')
    figure4.vlines(length90, 0.98, 1.08, linestyles="dashed", colors="k")
    figure5.plot(x1, dataSet, drawstyle='steps-post', color='b')
    figure5.plot(x1, yFit, color='r')
    figure5.plot(x1, YCIlower, color='b')
    figure5.plot(x1, YCIupper, color='b')
    figure5.vlines(length90, 0.98, 1.08, linestyles="dashed", colors="k")


def error(val1, val2):
    Err = abs((val2 - val1) / val2)
    print("Error =", Err)


def eq13(dataSet, yFit, fInt, lInt, dataSetSum, yFitSum):
    print("Eq. 13")
    R2 = dataSet[fInt - 1] * (lInt - fInt) - dataSetSum
    print("R2 =", R2)  # Real data
    R1 = yFit[fInt - 1] * (lInt - fInt) - yFitSum
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq14(dataSet, yFit, fInt, lInt, dataSetSum, yFitSum):
    print("Eq. 14")
    R2 = dataSetSum / (dataSet[fInt - 1] * (lInt - fInt))
    print("R2 =", R2)  # Real data
    R1 = yFitSum / (yFit[fInt - 1] * (lInt - fInt))
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq15(dataSet, yFit, fInt, lInt):
    print("Eq. 15")
    tempSum = 0
    for i in range((fInt - 1), lInt):
        tempSum += (dataSet[fInt - 1] - dataSet[i])
    R2 = tempSum / (dataSet[fInt] * (lInt - fInt))
    print("R2 =", R2)  # Real data
    tempSum = 0
    for i in range((fInt - 1), lInt):
        tempSum += (yFit[fInt - 1] - yFit[i])
    R1 = tempSum / (yFit[fInt] * (lInt - fInt))
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq17(fInt, lInt, dataSetSum, yFitSum):
    print("Eq. 17")
    R2 = dataSetSum / (lInt - fInt)
    print("R2 =", R2)  # Real data
    R1 = yFitSum / (lInt - fInt)
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq18(dataSet, yFit, fInt, lInt, dataSetSum, yFitSum):
    print("Eq. 18")
    R2 = (dataSet[fInt - 1] * (lInt - fInt) - dataSetSum) / (lInt - fInt)
    print("R2 =", R2)  # Real data
    R1 = (yFit[fInt - 1] * (lInt - fInt) - yFitSum) / (lInt - fInt)
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq16(dataSet, yFit, lInt, tdOne, tdTwo):
    print("Eq. 16")
    tempSum = 0
    for i in range((tdOne - 1), lInt):
        tempSum += dataSet[i]
    R2 = tempSum - dataSet[tdOne] * (lInt - tdOne)
    print("R2 =", R2)  # Real data
    tempSum = 0
    for i in range((tdTwo - 1), lInt):
        tempSum += yFit[i]
    R1 = tempSum - yFit[tdTwo] * (lInt - tdTwo)
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def eq19(dataSet, yFit, fInt, lInt, tdOne, tdTwo):
    print("Eq. 19")
    tempSum = 0
    for i in range((fInt - 1), tdOne):
        tempSum += dataSet[i]
    R2 = 0.5 * tempSum / (tdOne - fInt)
    tempSum = 0
    for i in range((tdOne - 1), lInt):
        tempSum += dataSet[i]
    R2 = R2 + (1 - 0.5) * tempSum / (lInt - tdOne)
    print("R2 =", R2)  # Real data
    tempSum = 0
    for i in range((fInt - 1), tdTwo):
        tempSum += yFit[i]
    R1 = 0.5 * tempSum / (tdTwo - fInt)
    tempSum = 0
    for i in range((tdTwo - 1), lInt):
        tempSum += yFit[i]
    R1 = R1 + (1 - 0.5) * tempSum / (lInt - tdTwo)
    print("R1 =", R1)  # Predicted
    error(R1, R2)


def metrics(dataSet, length, length90, yFit):
    # Predicted Metrics
    # Assuming th=n90+1 (our first predicted interval), tr=n(our last predicted interval)
    print("Metrics")
    th = length90 + 1
    tr = length

    # Eq. 12
    print("Eq. 12")
    tVecSum = 0
    for i in range((th - 1), tr):
        tVecSum += dataSet[i]
    print("R2 =", tVecSum)  # Real data
    YFitListSum = 0
    for i in range((th - 1), tr):
        YFitListSum += yFit[i]
    print("R1 =", YFitListSum)  # Predicted
    error(YFitListSum, tVecSum)

    # Eq. 13
    eq13(dataSet, yFit, th, tr, tVecSum, YFitListSum)

    # Eq. 14
    eq14(dataSet, yFit, th, tr, tVecSum, YFitListSum)

    # Eq. 15
    eq15(dataSet, yFit, th, tr)

    # Eq. 17
    eq17(th, tr, tVecSum, YFitListSum)

    # Eq. 18
    eq18(dataSet, yFit, th, tr, tVecSum, YFitListSum)

    # Metrics with td - We can't calculate them with only predicted data
    # Assuming th=1 (our first know interval), tr=n(our last predicted interval)
    print("Metrics with td")
    td1 = dataSet.index(min(dataSet))
    print("td1 =", (td1 + 1))
    td2 = yFit.index(min(yFit))
    print("td2 =", (td2 + 1))

    # Eq. 16
    eq16(dataSet, yFit, tr, td1, td2)

    # Eq. 19
    eq19(dataSet, yFit, th, tr, td1, td2)


def model(data, len, len90, equation, fig1, fig2, fig3, fig4, fig5):
    # Prediction
    YFitList = prediction(data, len, len90, equation, fig1, fig2)

    # Measure of the quality of the fit (page 60)
    S2 = quality(data, len, len90, YFitList)

    # Confidence Intervals
    confidence(data, len, len90, fig3, fig4, fig5, YFitList, S2)

    # METRICS
    metrics(data, len, len90, YFitList)


# Number of graphs
models = 2
figures = models * 5 + 1
figure, axis = plt.subplots(figures, 1)
figure.set_size_inches(8, (6 * figures))

# Data sets
# Nov 1973- Mar 1975 Recession
tVec = [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.9, 100.9, 100.9, 100.9, 100.9, 100.4, 99.7, 99.2, 98.7, 98.4, 98.1, 98.3, 98.2, 98.5, 99.0, 99.1, 99.5, 99.7, 100.1, 100.8, 101.2, 101.5, 101.8, 101.8, 101.9, 102.1, 102.3, 102.5, 102.6, 103.0, 103.3, 103.6, 104.0, 104.5, 104.9, 105.4, 105.9, 106.3, 106.6, 107.2, 107.6]
# Jan 1980 - Jul 1980 Recession
# tVec=[100.0,100.1,100.2,100.1,99.6,99.2,98.9,99.2,99.3,99.7,99.9,100.1,100.3,100.3,100.4,100.5,100.5,100.8,100.9,100.8,100.7,100.6,100.4,100.1,99.7,99.7,99.6,99.3,99.2,99.0,98.6,98.4,98.2,97.9,97.8,97.7,98.0,97.9,98.1,98.4,98.7,99.1,99.6,99.2,100.5,100.8,101.2,101.6]
# Jul 1981 - Nov 1982 Recession
# tVec=[100.0,100.0,99.9,99.8,99.5,99.2,98.9,98.9,98.7,98.4,98.4,98.1,97.7,97.6,97.4,97.1,96.9,96.9,97.1,97.1,97.3,97.6,97.9,98.3,98.7,98.4,99.6,99.9,100.3,100.7,101.2,101.7,102.0,102.4,102.7,103.1,103.5,103.7,104.1,104.4,104.8,104.9,105.2,105.3,105.7,105.9,106.2,106.4]
# Jul 1990 - Mar 191 Recession
# tVec=[100.0,99.8,99.7,99.6,99.5,99.4,99.3,99.0,98.9,98.7,98.6,98.6,98.6,98.6,98.6,98.7,98.6,98.6,98.7,98.6,98.7,98.8,98.9,99.0,99.0,99.2,99.2,99.4,99.5,99.7,100.0,100.2,100.1,100.4,100.6,100.8,101.1,101.2,101.4,101.7,101.9,102.2,102.5,102.6,103.1,103.4,103.7,104.0]
# Mar 2001 - Nov 2001 Recession
# tVec=[100,99.8,99.8,99.7,99.6,99.4,99.3,99,98.8,98.7,98.6,98.4,98.4,98.4,98.4,98.4,98.3,98.3,98.3,98.4,98.4,98.2,98.3,98.2,98,98,98,98,98,98,98.1,98.2,98.2,98.3,98.4,98.5,98.7,98.9,99.1,99.2,99.2,99.3,99.4,99.7,99.8,99.9,100,100.1]
# Dec 2007 - Jun 2009 Recession
# tVec=[100,100,100,99.9,99.7,99.6,99.5,99.3,99.1,98.8,98.4,97.9,97.4,96.8,96.3,95.7,95.2,94.9,94.6,94.3,94.2,94,93.9,93.8,93.7,93.7,93.7,93.8,94,94.4,94.2,94.2,94.2,94.1,94.3,94.4,94.5,94.5,94.7,94.9,95.1,95.1,95.2,95.2,95.3,95.4,95.5,95.6]
# Covid-19 Recession
# tVec=[1,0.862908,0.881689,0.913816,0.925259,0.935753,0.9405,0.945008,0.946758,0.94473,0.946274,0.949828,0.955032,0.956815,0.96068,0.966315,0.966315,0.96797693,0.969251284,0.974821656,0.976201111]
for i, point in enumerate(tVec):
    tVec[i] = (point / 100)

# Data set Plot
n = len(tVec)
print("n =", n)
n90 = math.floor(n * 0.9)
print("n90 =", n90)
x1 = range(0, n)
axis[0].plot(x1, tVec, drawstyle='steps-post', color='b')
axis[0].vlines(n90, 0.98, 1.08, linestyles="dashed", colors="k")

# Quadratic Model
print("\nQuadratic Model")


def sse(x):
    a = x[0]
    b = x[1]
    g = x[2]
    sseSum = 0
    for i in range(0, n90):
        sseSum += math.pow((tVec[i] - (a + b * i + g * i * i)), 2)
    return sseSum


def constraint(x):
    return x[1] + 2 * math.sqrt(x[0] * x[2])


bnds = ((0, None), (None, 0), (0, None))
LSfitRules = estimate(sse, [1, 0, 0], constraint, bnds)


def b2(t):
    return LSfitRules.x[0] + LSfitRules.x[1] * t + LSfitRules.x[2] * t * t


model(tVec, n, n90, b2, axis[1], axis[2], axis[3], axis[4], axis[5])

# Competing Risk
print("\nComputing Risk")


def sse(x):
    a = x[0]
    b = x[1]
    g = x[2]
    sseSum = 0
    for i in range(0, n):
        sseSum += math.pow(((a / (1 + b * i) + 2 * g * i) - tVec[i]), 2)
    return sseSum


def constraint(x):
    return x[0] * x[1] / 2 - x[2]


bnds = ((0, None), (0, None), (0, None))
LSfitRules = estimate(sse, [1, 0.1, 0.01], constraint, bnds)


def b2(t):
    return LSfitRules.x[0] / (1 + LSfitRules.x[1] * t) + 2 * LSfitRules.x[2] * t


model(tVec, n, n90, b2, axis[6], axis[7], axis[8], axis[9], axis[10])

# Must be at bottom
plt.show()