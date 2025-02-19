# Final Projects for Laboratory of Computational Physics

In each of the branches of this repo you find all the necessary to complete your final project.
In particular the file Project.ipynb describes the projects and provides guidance to its development.
Other files could be present if needed.

Each branch is named after the group of students a given project is assigned to.
The groups compositions are listed [here](https://docs.google.com/spreadsheets/d/124iLKfGEpxT33WgRqtuJqsoAjADjbarMWn5xxClTYEw/edit?gid=1442920615#gid=1442920615)

Students are supposed to work together to produce a short report on the assigned task. The preferred format for the latter is a jupyter notebook, with the proper description, the code implemented for the purpose and the actual results (plots, tables, etc.). The notebook has to be delivered with all the cells executed and should live in a GitHib repository. There is no need to make a pull request to the central repository.

### Computing Resources

A Virtual Machine within [CloudVeneto](http://cloudveneto.it/) can be created for each group. Note that, by default, they are not. For some projects though, large datasets are needed, in those cases a VM has been (are being) created to store those files. Refer to ClouldInstructions.md for the steps to take in order to use those resources.

Alternatively, students can use [colab](https://colab.research.google.com/) (for which though no instructions are provided here).

# Project description
## Trending Asset into Mean Reverting Asset

Trending assets are those whose prices or returns display a sustained movement in one direction (upward or downward) over time. Trending assets are typically non-stationary, meaning their statistical properties (e.g., mean, variance) evolve over time. This poses challenges for many standard financial analyses, which assume stationarity.

In this project our task has been to transform a trending financial asset into one that exhibits mean-reverting behavior through mathematical or statistical normalization techniques.

In particular we based our analysis on two different techniques:

1) Fitting a subtracting:

a) We implemented the Mann-Kendall test to assess the regions of monotonic trend in a given timeframe

The M-K test is a non parametric test to assess monotonic trends.
It is a hypothesis test:


*   H<sub>0</sub>: There is no monotonic trend in the data. The observations x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub> are independent and randomly ordered over time.
*   H<sub>a</sub>:There is a monotonic trend in the data.

The test computes the difference between each pair of data points and counts how many of these differences are positive or negative, so it evaluates if there are more increasing or decreasing values in the data. When applied to a timeseries the statistic of the test is defined as:

$$
S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn} (x_j - x_i)
$$
<br>
where:
$$
\text{sgn} (x_j - x_i) =
\begin{cases}
+1 & \text{if } (x_j - x_i) > 0 \\
0 & \text{if }(x_j - x_i) = 0 \\
-1 & \text{if }(x_j - x_i) < 0
\end{cases}
$$
<br>
and n is the sample size.
To evaluate the significance of 𝑆, the 𝑍 statistic is computed since for $n \geq 10$, S can be approximated by a normal distribution, allowing the use of a z-score:
<br>

$$
Z =
\begin{cases}
\frac{(S - 1)}{\sigma_s} & \text{if } S > 0 \\
0 & \text{if } S = 0 \\
\frac{(S + 1)}{\sigma_s} & \text{if } S < 0
\end{cases}
$$
<br>
where:
$$
\sigma_s = \sqrt{\frac{n(n-1)(2n+5) - \sum_{j=1}^{q} t_j (t_j - 1)(2t_j + 5)}{18}}
$$
<BR>
in which the first term of the difference is the variance of 𝑆 assuming no ties, while the sum accounts for ties, reducing the variance, since tied observations contribute less to a trend; *q* is the number of values that have ties, t<sub>j</sub> how many times the j-th value is repeated.<br>

If Z<Z<sub>α/2</sub> with a significance level α, then there doesn’t exist trend in the timeseries. (https://doi.org/10.2307/1907187, doi:10.1017/S0020268100013019)

b) In the trending regions we implemented a linear regression and subtracted the corresponding fitted line

2) Rolling mean:
a) We analysed the spectrum to identify a relevant frequency for trend -> a relevant timewindow
b) We apply a rolling average to the data with this timewindow and subtract the new timeserie to the original one

![An example of trending asset](Image/closing_prices_plot.png "An example of trending asset")


This project has been realized in collaboration and under the supervision of the company (XSOR Capital) [https://www.xsorcapital.com].

In particular we thank (Nicole Zattarin) [nicole.zattarin@xsorcapital.com] for her guidance.
