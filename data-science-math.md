### 函数与极限

#### 初等函数

常数函数 指数函数 幂函数 对数函数 三角函数 反三角函数

两个重要极限

间断点

第一类间断点 可去间断点,跳跃间断点

第二类间断点 

### 导数与微分

导数

微分

### 拟合(最小二乘法)

#### 单变量，线性拟合

拟合的核心在于我们如何去找到一个尽可能经过已知离散点的曲线.

一般如果是使用线性拟合就使用多项式作为基函数: 

    $\phi = \sum\limits_{i=0}^m a_i\phi_i(x_k)$,$(\phi_0=1,\phi_1=x,\phi_2=x^2.....)$

通过调节$a_i$来进行拟合(n为已知数据点数,m为我们确定的拟合次数).

如果是非线性拟合的话,可以使用$\phi=ae^{bx}+c$

通过$dis = \sum\limits_{k=0}^n(y_i - \sum\limits_{i=1}^ma_i\phi_i(x_k))^2$来衡量已知数据点和预测数据点的距离.　

样本索引为0...n，n为样本数-1

即拟合的目的是为了找到一个$\phi满足arg\min\limits_{\phi\in M} dis$ ($M=\sum\limits_{i=0}^ma_i\phi_i(x) ~~~a_i\in R$)M为基函数张成的空间.

机器学习中即是对该距离(loss)进行梯度下降,通过逼近的方式求到最优解,<mark>不用解析法的原因是由于其变量过多难以求解</mark>,而最小二乘法因为数量比较少,可以直接进行解析法求得

对dis进行求导得 $\partial dis/\partial a_j = -2\sum\limits_{k=0}^n(y_i-\sum\limits_{i=0}^ma_i\phi_i(x_k))\phi_j(x_k)$   j=0,1,..m

进行展开得 $\partial dis /\partial a_i = -2\sum\limits_{k=0}^ny_i\phi_j(x_k) +2\sum\limits_{k=0}^na_i\sum\limits_{i=0}^m\phi_i(x_k)\phi_j(x_k) $   j=0,1,2..m

令导数为0得 $\sum\limits_{i=0}^ma_i\sum\limits_{k=0}^n\phi_i(x_k)\phi_j(x_k) = \sum\limits_{k=0}^ny_i\phi_j(x_k)$  j=0,1,2..n<mark>为了求解这个方程将其转为为矩阵形式</mark>

等式右边可化为

$$
\begin{bmatrix}{\phi_j}&{\phi_j}&{\cdots}&{\phi_j}\end{bmatrix} * 
\begin{bmatrix}{y_o}\\{y_1}\\\vdots\\{y_n}\end{bmatrix}
$$

这其实也就是向量內积,故又将上式定义为$(\phi_j, y)$

同理等式左边得

$$
\begin{bmatrix}({\phi_0,\phi_j})&({\phi_1(x),\phi_j})&\cdots&({\phi_m,\phi_j})\end{bmatrix} *
\begin{bmatrix}a_0\\a_1\\\vdots\\a_m\end{bmatrix}
$$

于是对一个确定j有

$$
\begin{bmatrix}({\phi_0,\phi_j})&({\phi_1(x),\phi_j})&\cdots&({\phi_m,\phi_j})\end{bmatrix} *
\begin{bmatrix}a_0\\a_1\\\vdots\\a_m\end{bmatrix} = 
\begin{bmatrix}(\phi_0,{y})\\(\phi_1,{y})\\\vdots\\(\phi_m,y)\end{bmatrix}
$$

再进行拓展为

$$
\begin{bmatrix}({\phi_0,\phi_0})&({\phi_1,\phi_0})&\cdots&({\phi_m,\phi_0})
\\({\phi_0,\phi_1})&({\phi_1,\phi_1})&\cdots&({\phi_m,\phi_1})
\\\vdots&\vdots&\cdots&\vdots\\({\phi_0,\phi_m})&({\phi_1,\phi_m})
&\cdots&({\phi_n,\phi_m})
\end{bmatrix} *
\begin{bmatrix}a_0\\a_1\\\vdots\\a_m\\\end{bmatrix} = 
\begin{bmatrix}(\phi_0,{y})\\(\phi_1,{y})\\\vdots\\(\phi_m,y)\end{bmatrix}
$$

对该方程组进行求解即可得到$a_0,a_...a_m$

法方程组

$y^{'}=x\theta$  y'为一个列向量 

可将上边的平方差函数写为矩阵形式$dis=(y-x\theta)^T * (y-x\theta) = y^Ty-y^Tx\theta - \theta^Tx^Ty+\theta^Tx^Tx\theta$

求导得 $\partial y/ \partial \theta = -2y^Tx+2x^Tx\theta$   (这里的矩阵求导可以去看矩阵求导术)

然后令其为0 得  $y^Tx=x^Tx\theta$

然后有个地方需要注意就是 因为无法确定x是否有逆,但是$x^Tx$必有逆,所以就将$x^Tx$视为整体,最后得$(x^Tx)^{-1}y^Tx=\theta$

#### 非线性拟合

非线性拟合的关键就是转换为线性拟合

#### 多变量拟合

上面是单变量的情况（只有一个特征）

多变量的话: $\phi(x)=a_0+a_1x_1+a_2x_2+...+a_mx_m$  样本有m个特征

 平方误差函数有　$l=\sum\limits_{i=0}^n(y_i-(a_0+a_1x_{1i}+a_2x_{2i}+...+a_mx_{mi}))^2 $

样本索引为0...n，n为样本数-1

求导得　$\partial l/\partial a_j = -2\sum\limits_{i=0}^n(y_i-(a_0+a_1x_{1i}+a_2x_{2i}+...+a_mx_{mi}))x_{ji}~~~~j=1,2..m$

$~~~~~~~~~~~~~\partial l/\partial a_j = -2\sum\limits_{i=0}^n(y_i-(a_0+a_1x_{1i}+a_2x_{2i}+...+a_mx_{mi}))~~~j=0$

有法方程组

               $\sum\limits_{i=0}^n(a_0+a_1x_{1i}+a_2x_{2i}+..+a_mx_{mi}) = \sum\limits_{i=0}^ny_i$

               $\sum\limits_{i=0}^n(a_0+a_1x_{1i}+a_2x_{2i}+..+a_mx_{mi})x_{ji} = \sum\limits_{i=0}^nx_{ji}y_i ~~~j=1,2,..m$

 $\begin{bmatrix}n+1&\sum\limits_{i=0}^nx_{1i}&\cdots&\sum\limits_{i=0}^nx_{mi}\\ \sum\limits_{i=0}^nx_{1i}&\sum\limits_{i=0}^nx_{1i}^2&\cdots&\sum\limits_{i=0}^nx_{1i}x_{mi}\\ \sum\limits_{i=0}^nx_{2i}&\sum\limits_{i=0}^nx_{1i}x_{2i}&\cdots&\sum\limits_{i=0}^nx_{mi}x_{2i}\\\vdots\\\sum\limits_{i=0}^nx_{mi}&\cdots&\cdots&\sum\limits_{i=0}^nx_{mi}^2\end{bmatrix} * \begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\end{bmatrix} = \begin{bmatrix}\sum\limits_{i=0}^ny_i\\\sum\limits_{i=0}^nx_{1i}y_i\\\sum\limits_{i=0}^nx_{2i}y_i\\\vdots\\\sum\limits_{i=0}^nx_{mi}y_i\end{bmatrix}$

          

### 判别分析

判别分析主要就是用来进行分类的

假设目前有两蔟数据，对一个点(二维)进行

最简单的就是我们用预测点到各蔟数据的平均点的距离来衡量该点与该类别的近似程度

但是如果单纯依靠均值来进行判断，很不准确，因此我们一般使用马氏距离来计算距离，规避了数据的单位影响

因此有以下判别式

$W(x) = d^2(x,\pi_1) - d^2(x,\pi_2)$　这个上标2不是指平方，应该是表示这个距离的计算是二次的

$\begin{cases}W(x)>0~~x\in\pi_2\\W(x)<0~~x\in\pi_1\end{cases}$

如果是等于０，则判定为哪个类都可以

如果$W(x)>0$，则预测点距$\pi_1$的距离更远，则它就属于$\pi_2$

反之，就离$\pi_2$更远，属于$\pi_1$

继续对上述函数进行推导:

$Ｗ(x) = d^2(x,\pi_1) - d^2(x,\pi_2) $

$= (x-\mu_1)^{'}\sum_1^{-1}(x-\mu_1)-(x-\mu_2)^{'}\sum_2^{-1}(x-\mu_2)$

给出一个假设:$\sum_1^{-1}=\sum_2^{-1}$　该假设不一定成立，但是在工程中进行假设后预测效果尚可故一般都默认该假设成立

$= x^{'}\sum_1^{-1}x-\mu^{'}_1\sum_1^{-1}x-x^{'}\sum_1^{-1}\mu_1 +\mu^{'}_1\sum_1^{-1}\mu_1 -x^{'}\sum_2^{-1}x+\mu_2^{'}\sum_2^{-1}x+x^{'}\sum_2^{-1}\mu_2-\mu^{'}_2\sum_2^{-1}\mu_2$

$=\mu_2^{'}\sum_2^{-1}x-\mu_1^{'}\sum_1^{-1}x+x^{'}\sum_2^{-1}\mu_2-x^{'}\sum_1^{-}\mu_1+\mu_1^{'}\sum_1^{-}\mu_1-\mu_2^{'}\sum_2^{-1}\mu_2$

因为此处$\mu_1$或者$\mu_2$都是一个$n*1$的列向量（假设原数据有n个特征），则$\sum$为$n*n$的矩阵,x此处指预测点，因其只有一个故训练集为$n*1$，故上述式子的第三和第四，第一和第二乘出来都是标量，可以进行转置后合并，<mark>（如果不只一个点呢）</mark>

$=2\mu_2^{'}\sum_2^{-1}x-2\mu_1^{'}\sum_1^{-1}x+\mu_1^{'}\sum_1^{-1}\mu_1-\mu_2^{'}\sum_2^{-1}\mu_2$

$＝(2\mu_2^{'}-2\mu_1^{'})\sum_1^{-1}x+\mu_1^{'}\sum_1^{-1}\mu_1-\mu_2^{'}\sum_2^{-1}\mu_2$

$＝(2\mu_2^{'}-2\mu_1^{'})\sum_1^{-1}x+(\mu_1^{'}-\mu_2^{'})\sum_1^{-1}(\mu_1+\mu_2)$

$=(\mu_2^{'}-\mu_1^{'})\sum_1^{-1}(2x-\mu_1-\mu_2)$

$=2(\mu_2^{'}-\mu_1^{'})\sum_1^{-1}[x-(\mu_1+\mu_2)/2]$

令$a^{'}=(\mu_1^{'}-\mu_2^{'})\sum_1^{-1}$ 即$a=\sum_1^{-1}(\mu_1-\mu_2)$

$=-2a^{'}(x-\overline\mu)$

又因为$\sum^{-1}=\sigma^{-1}$

又可写为$W(x)=-2\sigma^{-1}(\mu_1-\mu_2)(x-\overline\mu)$

当数据为一维时:$\sigma^{-1}为标量且大于等于0$

如果$\mu_1>\mu_2$

则有$\begin{cases}x-\overline\mu>0 ~~x\in\pi_1\\x-\overline\mu<0~~x\in\pi_2\end{cases}$

反之，则互换

所以用$h(x)=x-\overline\mu$做判别函数也可以

当数据为多维的时候，此时$\sigma^{-1}$不一定各个元素都大于０，因此上述$h(x)$用不了，还是得用W(x)

其中，协方差矩阵用的是联合协方差矩阵$S=((n_1-1)S_1+(n_2-1)S_2)/(n_1+n_2-2)$

$S_1$和$S_2$分别为蔟１和蔟２的协方差矩阵，<mark>用联合协方差矩阵作为对数据总体的协方差矩阵的无偏估计</mark>

如果在某种情况下$\sum_1=\sum_2$不成立

则如果x只有单个特征

则$W(x)$与０的关系最后会化简为$|(x-\mu_1)/\sigma_1|$与$(x-\mu_2)/\sigma_2|$的大小关系(直接由马氏距离差推出)，由于存在绝对值，因此会有多个判别界线$(在x>\mu_1,\mu_2<x<\mu_1,x<\mu_2)(\mu_1>\mu_2的情况)$在不同区间内各有一个判别线（点）

而如果是多维的话，此时的判别称为二次判别

#### 多类判别

上述都是二分类的情况，而如果是多类的话，则通过计算各个类中心点到预测点之间的的距离找出最近的蔟，即为判别类

同样可假设$\sum_1=\sum_2=...=\sum_i$

则有$d^2(x,\pi_i)=(x-\mu_i)^{'}\sum^{-1}(x-\mu_i)\\=x^{'}\sum^{-1}x-2x^{'}\sum\mu_i+\mu_i^{-1}\sum\mu_i\\=x^{'}\sum^{-1}x-2(I'_ix+c_i)$

$I_i=\sum^{-1}\mu_i$     $c_i=-1/2\mu_i^{-1}\sum\mu_i$   i=1,2..k

式子中第一项可以看作是常数，故进行比较第二项即可

若满足$\max(I_i^{'}x+c_i)　　则有x\in\pi_i $

#### Summary

如果各组样本容量比较小，选择线性判别函数比较好，使用联合协方差来估计数据整个分布的协方差

如果数据比较多，用二次判别函数，此时用到的协方差就是精确估计(计算)的

### 聚类分析

相似性度量:距离,相似系数

距离又可分为:明氏距离,马式距离

相似系数:夹角余弦,相关系数

#### 距离

距离度量基本性质:

1. 非负性 $d(x,y)\ge0$

2. 同一性 d(x,y)=0仅当x=y

3. 对称性 d(x,y)=d(y,x)

4. 直递性 $d(x,y)\le d(x,z)+d(z,y)$

明考夫斯基距离 $d(x,y) = [\sum\limits_{i=1}^p|x_y-y_i|^q]^{1/q}$ 其实就是范数

q =1时,曼哈顿距离,差的绝对值之和

q=2时,欧式距离,差的平方和

一般在计算距离的时候,会先进行标准化,但因为标准化需要计算方差,所以有时候也用归一化$((x-x_{min})/(x_{max}-x_{min}))$

如果是名义变量,即非数值形变量,可使用名义变量的一种距离定义为

$d=m_2/(m_1+m_2)$ $m_2$是不配合变量,$m_1$是配合变量,变量取值相同称为配合变量

马氏距离也可以

#### 相似系数

相似系数应满足的条件

$c_{ij}=+1,-1$ 当且仅当$x_i=ax_j+b$

$|c_{ij}| \le 1$

$|c_{ij}=c_{ji}|$

两个向量之间的相似系数用夹角余弦表示

$c_{ij}(1)=\sum\limits_{k=1}^n x_{ki}x_{kj}/[(\sum\limits_{k=1}^nx_{ki}^2)(\sum\limits_{k=1}^nx_{kj}^2)]^{1/2}$

$cos(\theta)= x^{'}y/(||x||~||y||)$

#### 系统聚类法 (层次聚类)

分为自顶向下和(分割)自底向上两种(聚集)

计算蔟间的距离 (蔟间!!)

1. 最短距离

2. 最长距离

3. 平均距离,计算两个蔟对应点之间的距离的均值

4. 重心法,计算两个蔟中心点间的差的平方 $d=(\overline{x_k}-\overline{x_L})^2$

5. 离差平方和 $d=n_kn_l/(n_k+n_l)~*~d_{重心}$ $n_k$为该蔟包含的元素数量,解决蔟间个数不平衡的问题

先将各个数据视为一个蔟，然后与距离自己最近的蔟合并（如果有独立的，则保持单个为一类），重复该步骤，直到达到设定的分类类别数
