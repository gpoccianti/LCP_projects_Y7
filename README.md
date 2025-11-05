# Final Projects for Laboratory of Computational Physics

In each of the branches of this repo, you will find all the necessary materials to complete your final project. In particular, the file `Project.ipynb` describes the projects and provides guidance for their development. Other files may be present if needed.

Each branch is named after the group of students a given project is assigned to. The group compositions are listed [here](https://docs.google.com/spreadsheets/d/124iLKfGEpxT33WgRqtuJqsoAjADjbarMWn5xxClTYEw/edit?gid=1442920615#gid=1442920615).

Students are expected to work together to produce a short report on the assigned task. The preferred format for the report is a Jupyter Notebook, containing the proper description, the implemented code, and the actual results (plots, tables, etc.). The notebook must be delivered with all cells executed and should reside in a GitHub repository. There is no need to make a pull request to the central repository.

## Computing Resources

A Virtual Machine within [CloudVeneto](http://cloudveneto.it/) can be created for each group. Note that, by default, they are not created. However, for some projects that require large datasets, a VM has been (or is being) created to store those files. Refer to `CloudInstructions.md` for the steps to use these resources.

Alternatively, students can use [Google Colab](https://colab.research.google.com/) (though no instructions are provided here).

# Project Description
## Trending Asset into Mean Reverting Asset

Trending assets are those whose prices or returns display a sustained movement in one direction (upward or downward) over time. Such assets are typically non-stationary, meaning their statistical properties (e.g., mean, variance) evolve over time. This poses challenges for many standard financial analyses, which assume stationarity.

In this project, our task was to transform a trending financial asset into one that exhibits mean-reverting behavior through mathematical or statistical normalization techniques.

We based our analysis on two different techniques:

### 1) Fitting and Subtracting:

#### a) Mann-Kendall Test to Identify Trends

The Mann-Kendall (M-K) test is a non-parametric test used to assess monotonic trends. It is a hypothesis test:

- **H₀**: There is no monotonic trend in the data. The observations \( x_1, x_2, ..., x_n \) are independent and randomly ordered over time.
- **Hₐ**: There is a monotonic trend in the data.

The test computes the difference between each pair of data points and counts how many of these differences are positive or negative, evaluating whether there are more increasing or decreasing values in the dataset.

When applied to a time series, the test statistic \( S \) is defined as:

$$
S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn} (x_j - x_i)
$$

where:

$$
\text{sgn} (x_j - x_i) =
\begin{cases}
+1 & \text{if } (x_j - x_i) > 0 \\
0 & \text{if }(x_j - x_i) = 0 \\
-1 & \text{if }(x_j - x_i) < 0
\end{cases}
$$

For \( n \geq 10 \), \( S \) can be approximated by a normal distribution, allowing the use of a z-score:

$$
Z =
\begin{cases}
\frac{(S - 1)}{\sigma_s} & \text{if } S > 0 \\
0 & \text{if } S = 0 \\
\frac{(S + 1)}{\sigma_s} & \text{if } S < 0
\end{cases}
$$

where:

$$
\sigma_s = \sqrt{\frac{n(n-1)(2n+5) - \sum_{j=1}^{q} t_j (t_j - 1)(2t_j + 5)}{18}}
$$

Here, the first term in the numerator is the variance of \( S \) assuming no ties, while the summation accounts for ties, reducing the variance. \( q \) is the number of unique values that have ties, and \( t_j \) is the count of occurrences for each tied value.

If $ Z < Z_{\alpha/2} $ at a significance level \( \alpha \), then no significant trend exists in the time series.

(References: [https://doi.org/10.2307/1907187](https://doi.org/10.2307/1907187), [doi:10.1017/S0020268100013019](doi:10.1017/S0020268100013019))

#### b) Linear Regression on Trending Regions

For the identified trending regions, we applied a linear regression and subtracted the corresponding fitted line to remove the trend component.

### 2) Rolling Mean:

#### a) Spectral Analysis to Determine Time Window

We analyzed the spectrum of the time series to identify a relevant frequency indicative of a trend. This frequency helped define a meaningful time window for further processing.

#### b) Rolling Average and Subtraction

We applied a rolling average with the determined time window and subtracted the resulting smoothed time series from the original data.

![An example of a trending asset](Image/closing_prices_plot.png "An example of a trending asset")

## Acknowledgments

This project was realized in collaboration with and under the supervision of [XSOR Capital](https://www.xsorcapital.com).

In particular, we thank **Nicole Zattarin** ([nicole.zattarin@xsorcapital.com](mailto:nicole.zattarin@xsorcapital.com)) for her guidance.

