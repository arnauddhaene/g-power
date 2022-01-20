from typing import Callable, Any

import pandas as pd
import numpy as np
import pingouin as pg

import scipy.stats as sps

import streamlit as st

st.set_page_config(
    layout='centered',
    page_icon='https://static.macupdate.com/products/24037/l/gpower-logo.png?v=1588748934')

# Styles
with open('styles.css') as file:
    styles = file.read()
st.write(f'<style> {styles} </style>', unsafe_allow_html=True)

st.title('G*Power Online')

family, statistical = st.columns([1, 4])

test_families = ['Exact', 'F tests', 't tests', 'χ² tests', 'z tests']
family.selectbox('Test family', test_families, index=2)

statistical_tests = [
    'Correlation: Point biserial model',
    'Linear bivariate regression: One group, size of slope',
    'Linear bivariate regression: Two groups, difference between intercepts',
    'Linear bivariate regression: Two groups, difference between slopes',
    'Linear multiple regression: Fixed model, single regression coefficient',
    'Means: Difference between two dependent means (matched pairs)',
    'Means: Difference between two independent means (two groups)',
    'Means: Difference from constant (one sample case)',
    'Means: Wilcoxon signed-rank test (matched pairs)',
    'Means: Wilcoxon signed-rank test (one sample case)',
    'Means: Wilcoxon-Mann-Whitney test (two groups)',
    'Generic t test']
statistical.selectbox('Statistical test', statistical_tests)

analysis_types = [
    'A priori: Compute required sample size - given α, power, and effect size',
    'Compromise: Compute implied α & power - given β/α ratio, sample size, and effect size',
    'Criterion: Compute required α - given power, effect size, and sample size',
    'Post hoc: Compute achieved power - given α, sample size, and effect size',
    'Sensitivity: Compute required effect size - given α, power, and sample size']
st.selectbox('Type of power analysis', analysis_types)

inputs, outputs = st.columns([3, 2])

inputs.write("""
             ### Input parameters
             """)

tails = inputs.selectbox('Tail(s)', ['One', 'Two'])
d = inputs.number_input('Effect size d', value=0.2)
alpha = inputs.number_input('α error prob', value=.05, min_value=1e-7, max_value=1. - 1e-7)
power = inputs.number_input('Power (1-β err prob)', value=.8, min_value=1e-7, max_value=1. - 1e-7)
# ratio = inputs.number_input('Allocation ratio N2/N1', value=1., min_value=0.01, max_value=100.)

if tails == 'Two':
    alternative = 'two-sided'
else:
    alternative = 'greater' if d > 0 else 'less'

outputs.write("""
             ### Output parameters
             """)

dtype_bible = {
    'Noncentrality parameter δ': 'float',
    'Critical t': 'float',
    'Df': 'int',
    'Sample size group 1': 'int',
    'Sample size group 2': 'int',
    'Total sample size': 'int',
    'Actual power': 'float'
}


def float_format(x: float) -> str:
    return f'{x:.7f}'


def int_format(x: int) -> str:
    return f'{x:.0f}'


def format_from_dtype(dtype: str) -> Callable[Any, str]:
    if dtype == 'int':
        return int_format
    elif dtype == 'float':
        return float_format


variables = dtype_bible.keys()
formatters = dict(zip(variables,
                  map(lambda r: format_from_dtype(dtype_bible[r]), variables)))


def format_row_wise(styler, formatter):
    for row, row_formatter in formatter.items():
        row_num = styler.index.get_loc(row)

        for col_num in range(len(styler.columns)):
            styler._display_funcs[(row_num, col_num)] = row_formatter
    return styler


if st.button('Calculate'):
    n = np.ceil(pg.power_ttest(d=d, power=power, alpha=alpha, alternative=alternative))
    
    df = pd.DataFrame({
        'Noncentrality parameter δ': d * np.sqrt(n / 2.),
        'Critical t': sps.t.ppf(
            1 - alpha / 2 if alternative == 'two-sided' else alpha,
            (n - 1) * 2) * (-1 if d > 0 else 1),
        'Df': (n - 1) * 2,
        'Sample size group 1': n,
        'Sample size group 2': n,
        'Total sample size': n * 2,
        'Actual power': pg.power_ttest(n=n, d=d, alpha=alpha, alternative=alternative)
    }, index=['Value'])
    
    df = df.astype(dtype=dtype_bible).transpose()
    
    outputs.dataframe(format_row_wise(df.style, formatters))
