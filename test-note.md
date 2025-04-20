In linear regression, we considered the problem
$$
\min_\beta \|y - X\beta\|^2
$$
If $X$ is singular, or if we wish to penalise large weights, then we can add a regularisation term. The updated problem takes the form
$$
\min_\beta \|y - X\beta\|^2 +\lambda\|\beta\|^2
$$
In this note, we will see how we can choose a good $\lambda$. We will also see a generalisation of Ridge regularisation: the Tikhonov regularisation.

# <span style="color:#295ABB">1. Ridge regularisation</span>
---
Let $X\in\mathbb R^{m\times n}$, $y\in\mathbb R^m$ and $\lambda>0$. Consider the problem
$$
\min_\beta \|y - X\beta\|^2 +\lambda\|\beta\|^2
$$
Using the Pythagorean theorem, we see that this is equivalent to finding the least squares solution of
$$
\begin{bmatrix}
X \\ \sqrt\lambda I
\end{bmatrix}\beta
= \begin{bmatrix}
y \\ 0
\end{bmatrix} \quad (1)
$$
This problem will have a unique solution, since the matrix on the left has linearly independent columns, no matter if $X$ is singular or not. If we use the normal equations on $(1)$, we get that
$$
\begin{aligned}
\left(\begin{bmatrix}
X^T & \sqrt\lambda I
\end{bmatrix}
\begin{bmatrix}
X \\ \sqrt\lambda I
\end{bmatrix}
\right)\hat\beta
&=
\begin{bmatrix}
X^T & \sqrt\lambda I
\end{bmatrix}
\begin{bmatrix}
y \\ 0
\end{bmatrix}
\\
(X^TX + \lambda I)\hat\beta
&=
X^Ty
\end{aligned}
$$
and hence:

>[!NOTE] **Theorem 1** (Ridge regression)
>Let $X$ be $m\times n$ and $y\in\mathbb R^m$.  Then the least squares solution of $(1)$ is
>$$
>\hat\beta = (X^TX + \lambda I)^{-1}X^Ty\quad (2)
>$$

# <span style="color:#295ABB">2. Ridge regression using the SVD</span>
---
For the problem
$$
\min_\beta \|y - X\beta\|^2 +\lambda\|\beta\|^2 \quad (3)
$$
how should we choose $\lambda$? One way is to use leave-one-out cross validation on a range of different $\lambda$s, and choose the one which gives the best $\text{RMSE}$. We may compute the coefficients using $(2)$. This is slow. Another approach is to use the QR-factorisation. If we let 
$$
Z_\lambda = 
\begin{bmatrix}
X \\ \sqrt\lambda I
\end{bmatrix}
$$
then the least squares solution to $(3)$ is
$$
\hat\beta = R^{-1}Q^Tb
$$
where $Z_\lambda = QR$. This does give an improvement in terms of speed, but we can do better. We first need the following result.

>[!NOTE] **Theorem 2** (Ridge regression using svd)
>Let $X = U\Sigma V^T$ be the SVD of $X$. Then the least squares solution of $(3)$ is
>$$
>\hat\beta = V(\Sigma^T\Sigma + \lambda I_n)^{-1}\Sigma U^Ty \quad (4)
>$$
>In addition, the singular values of $Z_\lambda^T$ are $\sqrt{\sigma_i^2+\lambda}$.

**Proof.** Consider the full SVD $X=U\Sigma V^T$, then $X^TX = V\Sigma^T\Sigma V^T$. The columns of $V$ are the eigenvectors of $X^TX$, but they are also the eigenvectors of $Z_\lambda^TZ_\lambda$, since
$$
Z_\lambda^TZ_\lambda = X^TX+\lambda I_n = V\Sigma^T\Sigma V^T +\lambda V^TVV^TV = V(\Sigma^T\Sigma + \lambda I_n)V^T
$$
We observe that the singular values of $Z_\lambda$ must be $\sqrt{\sigma_i^2+\lambda}$. Further, using the fact that $(AB)^{-1} = B^{-1}A^{-1}$ and $V^{-1}=V^T$, we get that
$$
(Z_\lambda^TZ_\lambda)^{-1}=(V(\Sigma^T\Sigma + \lambda I_n)V^T)^{-1} = V(\Sigma^T\Sigma + \lambda I_n)^{-1}V^T
$$
Putting this into $(2)$ yields
$$
\begin{aligned}
\hat\beta &=
V(\Sigma^T\Sigma + \lambda I_n)^{-1}V^TX^Ty \\
&= 
V(\Sigma^T\Sigma + \lambda I_n)^{-1}V^TV\Sigma U^Ty \\
&=
V(\Sigma^T\Sigma + \lambda I_n)^{-1}\Sigma U^Ty
\end{aligned}
$$
This establishes the result. $\blacksquare$

The next theorem shows that the least squares solution $\hat\beta$ of $(3)$ is just a linear combination of the columns of $V_r$ (the right singular vectors of $X$).

>[!NOTE] **Theorem 3** (Ridge regression and the right singular vectors)
>The least squares solution of $(3)$ is
>$$
>\hat\beta = V_rc_\lambda
>$$
>where $c_\lambda = (\Sigma_r + \lambda \Sigma_r^{-1})^{-1}U_r^Ty$ and $X = U_r\Sigma_r V_r^T$ is the reduced SVD of $X$. 

**Proof.** Notice that we may reformulate $(4)$ as 
$$
\begin{aligned}
\hat\beta 
&=
V(\Sigma^T\Sigma + \lambda \Sigma\Sigma^{-1})^{-1}\Sigma U^Ty \\
&=
V(\Sigma[\Sigma + \lambda \Sigma^{-1}])^{-1}\Sigma U^Ty \\
&=
V(\Sigma + \lambda \Sigma^{-1})^{-1}\Sigma^{-1}\Sigma U^Ty \\
&=
V(\Sigma + \lambda \Sigma^{-1})^{-1}U^Ty
\end{aligned}
$$
And if we use the reduced SVD $X = U_r\Sigma_r V_r^T$, we end up with
$$
\hat\beta = V_r(\Sigma_r + \lambda \Sigma_r^{-1})^{-1}U_r^Ty
$$
and the desired result follows immediately. $\blacksquare$

If we write $c_\lambda = \begin{bmatrix}c_{\lambda, 1},\dots,c_{\lambda, r}\end{bmatrix}^T$, we also see that 
$$
c_{\lambda, k} = \frac{u_k^Ty}{\sigma_k^2+\lambda/\sigma_k}
$$
where $\sigma_k$ is the $k$th singular value of $X$. The next theorem shows a similar result: the fitted values using the least squares solution are linear combinations of the $r$ first left singular values of $X$.

>[!NOTE] **Theorem 4** (Ridge regression and the left singular vectors)
>The fitted values of the least squares solution of $(3)$ is
>$$
\hat y = U_rd_\lambda
>$$
>where $d_\lambda = \Sigma_rc_\lambda$, $c_\lambda = (\Sigma_r + \lambda \Sigma_r^{-1})^{-1}U_r^Ty$ and $X = U_r\Sigma_r V_r^T$ is the reduced SVD of $X$

**Proof.** Using the preceding theorem, we get that the fitted values $\hat y$ are given by
$$
\hat y = X\hat\beta = XV_rc_\lambda = U_r\Sigma_r V_r^TV_rc_\lambda = U_r\Sigma_rc_\lambda = U_rd_\lambda
$$
where $d_\lambda = \Sigma_rc_\lambda$. $\blacksquare$

In the case of multiple responses, $Y\in\mathbb R^{m\times q}$, we get the following formulation
$$
\begin{aligned}
B_\lambda &= V_rC_\lambda \\
\hat Y &= U_rD_\lambda
\end{aligned}
$$
where $C_\lambda = (\Sigma_r + \lambda \Sigma_r^{-1})^{-1}U_r^TY$ and $D_\lambda = \Sigma_rC_\lambda$.


# <span style="color:#295ABB">3. Fast LooCV using Ridge regression</span>
---
Recall the leave-one-out result, where $(x^{(i)}, y^{(i)})$ be the $i$th fold (sample), and $(X^i, y^i)$ the remaining data

>[!NOTE] **Theorem 5** (Fast LooCV)
>Let $(X, y)$ be a data set. If $\{(x_i, y_i),\space(X^i, y^i)\}$ is the $i$th fold in a leave-one-out cross-validation and
>1. $\hat\beta$ is the OLS solution of $X\beta = y$
>2. $\hat\beta_i$ the OLS solution of $X^i\beta_i = y_i$
>3. $r_i = y_i-x_i\hat\beta$ the residuals of the $i$th fold using the full model
>
>Then the residual $r_{(i)} = y_i-x_i\hat\beta_i$ can be obtain the following way
>$$
>r_{(i)} = \frac{r_i}{1-h_i}
>$$
>Where $h_i = x_i(X^TX)^{-1}x_i^T$.

Note that $h_i$ is the $i$th element of $H = X(X^TX)^{-1}X^T$. Since $H$ is a projection matrix, then there must exist an orthogonal matrix $U$, where $\text{Col}(U) = \text{Col}(X)$, such that
$$
H = UU^T
$$
and hence we get the $h_i = u_iu_i^T$, where $u_i$ is the $i$th row of $U$. If we wish to include a constant term to, that is, we add a column of $1$s to $X$, then we get the following result.

>[!NOTE] **Theorem 6** (Fast LooCV with constant term)]
>If $X$ is centered and $\text{Col }X = \text{Col }U$, $U$ orthogonal columns, then the $i$th LooCV prediction residual is given by
>$$
>r_{(i)}=\frac{r_i}{1-h_i^*}
>$$
>where $h_i^* = u_iu_i^T + \tfrac{1}{m} = h_i + \tfrac{1}{m}$.

**Proof.** If $\text{rank }A = r$, then $U$ is $m\times r$. Since $X$ is centered, we must have that $X^Tu_0 = (u_0^TX)^T = \begin{bmatrix}0 & \dots & 0 \end{bmatrix}$. Since $X$ and $U$ span the same subspace, it must be the case that $U^Tu_0 = (u_0^TU)^T = \begin{bmatrix}0 & \dots & 0 \end{bmatrix}$, and hence
$$
\begin{bmatrix}u_0^T \\ U^T\end{bmatrix}\begin{bmatrix}u_0 & U\end{bmatrix}
= \begin{bmatrix}u_0^Tu_0 & u_0^TU \\ U^Tu_0 & U^TU\end{bmatrix}
= \begin{bmatrix}1 & 0 \\ 0 & I_r\end{bmatrix} = I_{r+1}
$$
which shows that $\begin{bmatrix}u_0 & U\end{bmatrix}$ has orthogonal columns. Since $u_0$ is just the ${\bf 1}$ vector scaled, we also have that
$$
\text{Col}(\begin{bmatrix}1 & X\end{bmatrix}) = 
\text{Col}(\begin{bmatrix}u_0 & U\end{bmatrix})
$$
Indeed
$$
H^* = \begin{bmatrix}u_0 & U\end{bmatrix}\begin{bmatrix}u_0^T \\ U^T\end{bmatrix} 
= u_0u_0^T + UU^T = \frac{1}{m}{\bf 1}_{m\times m}+UU^T
$$
and the $i$th row is $h_i^* = \frac{1}{m} + h_i$ where $h_i$ is the $i$th row of $H = UU^T$. The desired conclusion follows. $\blacksquare$

By definition, $r_{(i)} = y_i - \hat y_{i, -1}$, where $\hat y_{i, -1}$ is the fitted value using a model trained on $X$ where observation $i$ is left out. We have that
$$
r_{(i)} = y_i - \hat y_{i, -1}
$$
so
$$
y_i - \hat y_{i, -1} = y_i - (y_i - r_{(i)}) = \frac{r_i}{1-h_i^*} = \frac{y_i - \hat y_i}{1-h_i^*}
$$
which leads to the following definition.

>[!IMPORTANT] **Definition** PRESS
>Let $\hat y_{i, -1}$ denote the prediction of the $i$th row of $X$ when $x_i$ is left out. The predicted residual sum of squares is
>$$
>\text{PRESS} = \sum_{i=1}^m(y_i - \hat y_{i, -1})^2 = \sum_{i=1}^m\frac{(y_i - \hat y_i)^2}{(1 - h_i - 1/m)^2}
>$$

The values $h = [h_{ii}]$ on the diagonal of $H$ is known as the **leverage**. The following theorem states a fast way of calculating $h$.

>[!NOTE] **Theorem 7** (Leverage)
>The leverage values of $X$ can be found as follows,
>$$
>h = (U \odot U){\bf 1}
>$$
>where $\odot$ denotes element-wise multiplication.


# <span style="color:#295ABB">4. Tikhonov regularisation</span>
---
Consider the problem
$$
\min_\beta \|y - X\beta\|^2 +\lambda\|L\beta\|^2\quad (10)
$$
where $L$ is some matrix. What we are doing here is restricting the size of each individual component of $\beta$, or a linear combination of the components. Like before, this is equivalent of finding the least squares solution $\hat\beta$ of
$$
\begin{bmatrix}
X \\ \sqrt\lambda L
\end{bmatrix}\beta
= \begin{bmatrix}
y \\ 0
\end{bmatrix}
$$
If $L$ is invertible, we may write this as
$$
\begin{bmatrix}
XL^{-1} \\ \sqrt\lambda I
\end{bmatrix}L\beta
= \begin{bmatrix}
y \\ 0
\end{bmatrix}
$$
Setting $\beta^\star = L\beta$ and $X^\star = XL^{-1}$, we obtain
$$
\begin{bmatrix}
X^\star \\ \sqrt\lambda I
\end{bmatrix}\beta^\star
= \begin{bmatrix}
y \\ 0
\end{bmatrix}
$$
which is just the Ridge formulation $(3)$, that is
$$
\min_{\beta^\star} \|y - X^\star\beta^\star\|^2 +\lambda\|\beta^\star\|^2
$$
which we already know how to solve. To summarise:

>[!NOTE] **Theorem 7** (Tikhonov regression)
>Let $X$ be $m\times n$, $y\in\mathbb R^m$, $\beta>0$ and $L$ and $n\times n$ invertible matrix. Let $X^\star = XL^{-1}$, then the solution to $(10)$ is given by $\beta^\star = L\beta$ where $\beta^\star$ is the solution of
>$$
>\min_{\beta^\star} \|y - X^\star\beta^\star\|^2 +\lambda\|\beta^\star\|^2
>$$

This last theorem shows an inexpensive way of computing the leverages $h_\lambda$ and the $\text{PRESS}(\lambda)$ values of a Ridge regression problem.

>[!NOTE] **Theorem 8** (leverage and press for Ridge)
>Let $U_r\Sigma_r V_r^T$ be the SVD of $X$ with non-zero singular values $\sigma_1,\dots,\sigma_r$. The leverage $h_\lambda$ and the $\text{PRESS}(\lambda)$ values of the problem $\min_\beta \|y - X\beta\|^2 +\lambda\|\beta\|^2$ is given by
>$$
>\begin{aligned}
>&h_\lambda = (\Gamma_\lambda \odot \Gamma_\lambda){\bf 1} \\
>&\text{PRESS}(\lambda) = \|r_\lambda^*\|^2
>\end{aligned}
>$$
>where $\Gamma_\lambda = U_r\Sigma_r\Sigma_{(\lambda, r)}^{-1}$,  $\Sigma_{(r, \lambda)} = \text{diag}(\sigma_{\lambda, i})$, $\sigma_{\lambda, i} =\sqrt{\sigma_i^2+\lambda}$, $r_{\lambda,i}^* = \frac{r_{\lambda,i}}{1-h_{\lambda,i}-1/m}$, $r_{\lambda,i} = y_i - \hat y_{\lambda,i}$, and where $\hat y_\lambda$ is the fitted values of the full Ridge regression model.

**Proof.** As we saw fro theorem 2, if the SVD of $X$ is $U_r\Sigma_r V_r^T$ with (non-zero) singular values $\sigma_1,\dots,\sigma_r$, then the singular values of $Z_\lambda$ are $\sigma_{\lambda, i} =\sqrt{\sigma_i^2+\lambda}$, where we recall that
$$
Z_\lambda = 
\begin{bmatrix}
X \\ \sqrt\lambda I
\end{bmatrix}
$$
If $U_{(\lambda, r)}\Sigma_{(\lambda, r)}V_{(\lambda, r)}$ is the (reduced) SVD of $Z_\lambda$, it must be that $\Sigma_{(r, \lambda)} = \text{diag}(\sigma_{\lambda, i}),\space i=1,\dots,r$, and that $V_{(\lambda, r)} = V_r$. The associated left singular values are 
$$
\begin{aligned}
U_{(\lambda, r)} 
&= Z_\lambda V_r\Sigma_{(r, \lambda)}^{-1} \\
&= \begin{bmatrix}
XV_r\Sigma_{(r, \lambda)}^{-1} \\ \sqrt\lambda V_r\Sigma_{(r, \lambda)}^{-1}
\end{bmatrix} \\
&= \begin{bmatrix} 
U_r\Sigma_r V_r^TV_r\Sigma_{(r, \lambda)}^{-1} \\ \sqrt\lambda V_r\Sigma_{(r, \lambda)}^{-1}
\end{bmatrix} \\
&= \begin{bmatrix} 
U_r\Sigma_r\Sigma_{(r, \lambda)}^{-1} \\ \sqrt\lambda V_r\Sigma_{(r, \lambda)}^{-1}
\end{bmatrix} \\
&= \begin{bmatrix} 
\Gamma_\lambda \\ \sqrt\lambda V_r\Sigma_{(r, \lambda)}^{-1}
\end{bmatrix}
\end{aligned}
$$
where $\Gamma_\lambda = U_r\Sigma_r\Sigma_{(\lambda, r)}^{-1}$. The leverage is therefore given by
$$
h_\lambda = (\Gamma_\lambda\odot\Gamma_\lambda){\bf 1}
$$
and the $\text{PRESS}$ values are
$$
\text{PRESS}(\lambda) = \sum_{i=1}^m\frac{(y_i - \hat y_{\lambda,i})^2}{(1 - h_{\lambda,i}-1/m)^2}
$$
But since $r_{\lambda,i} = y_i - \hat y_{\lambda,i}$ and $r_{\lambda,i}^* = \frac{r_{\lambda,i}}{1-h_{\lambda,i}-1/m}$, we get that $\text{PRESS}(\lambda) = r_\lambda^{*T}r_\lambda^*$, or simply
$$
\text{PRESS}(\lambda) = \|r_\lambda^*\|^2
$$
which is the desired result. $\blacksquare$

As a last note, assume that there is some known prior knowledge about the coefficients $\beta$. Using the generalised Tikhonov regularisation, we may add a preference towards a prior $d$ (known or expected weights, or a prior solution). The formulation is as follows
$$
\min_\beta\|y-X\beta\|^2 + \lambda\|L\beta-d\|^2
$$
This is also closely related to Lavrentyev regularisation.