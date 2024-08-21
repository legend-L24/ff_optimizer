### 基于Zwanzig reweighting方案的物理性质优化

这一优化方案的核心是如下推导过程：

$$\left<A\right>_{H_1}=\frac{\int{A\left(r\right)\exp\left[-\beta U^{H_1}\left(r\right)\right]dr}}{Z_{H_1}}$$
$$=\frac{\int{A\exp\left[-\beta \left(U^{H_1}-U^{H_2}+U^{H_2}\right)\right]dr}}{Z_{H_1}}$$
$$=\frac{\int{A\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\exp\left(-\beta U^{H_2}\right)dr}}{Z_{H_1}}$$
$$=\frac{\int{A\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\exp\left(-\beta U^{H_2}\right)dr}}{Z_{H_2}}\frac{Z_{H_2}}{Z_{H_1}}$$
$$=\left<A\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\right>_{H_2}\frac{Z_{H_2}}{Z_{H_1}}$$

其中$Z_{H_1}=\int{\exp\left[-\beta U^{H_1}\left(r\right)\right]dr}$, $Z_{H_2}=\int{\exp\left[-\beta U^{H_2}\left(r\right)\right]dr}$

这表示我们可以用$H_2$系综下的平均值来估算$H_1$的系综平均。

另外，$\frac{Z_{H_1}}{Z_{H_2}}$可以进一步变换为：
$$\frac{Z_{H_1}}{Z_{H_2}}=\frac{\int{\exp\left(-\beta U^{H_1}\right)dr}}{Z_{H_2}}$$
$$=\frac{\int{\exp\left[-\beta \left(U^{H_1}-U^{H_2}+U^{H_2}\right)\right]dr}}{Z_{H_2}}$$
$$=\frac{\int{\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\exp\left(-\beta U^{H_2}\right)dr}}{Z_{H_2}}$$
$$=\left<\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\right>_{H_2}$$

两式结合有：
$$\left<A\right>_{H_1}=\left<\frac{\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]}{\left<\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\right>_{H_2}} A\right>_{H_2}=\sum^{H_2}_i {\frac{\exp\left[-\beta \left(U^{H_1}_i-U^{H_2}_i\right)\right]}{\left<\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\right>_{H_2}} A_i}$$

$\frac{\exp\left[-\beta \left(U^{H_1}_i-U^{H_2}_i\right)\right]}{\left<\exp\left[-\beta \left(U^{H_1}-U^{H_2}\right)\right]\right>_{H_2}}$可以看作是用$H_2$势函数采样来估计$H_1$采样时，构象$i$的权重。

`ReweightEstimator`是DMFF提供的、使用reweighting方法，以可微分的方式估计物理量的模块。使用这一模块的前提是实现若干采样与重算的功能。

### 液相采样函数实现（MD模拟）