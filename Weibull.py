from operator import truediv
from statistics import quantiles
from tokenize import Number
import streamlit as st
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt


st.title("Weibull")
col1, col2 = st.columns(2)
amostras_falhadas = dict()
amostras_censuradas = dict()
xmin = 0
xmax = 0
num_falhas = st.sidebar.slider("Quantas amostras com falhas?", 0, 15, 0)
num_censuradas = st.sidebar.slider(
    "Quantas amostras censuradas?", 0, 15, 0)
if num_falhas > 0:
    st.sidebar.write("Amostras Falhadas:")
    for i in range(num_falhas):
        amostras_falhadas[i] = st.sidebar.number_input(
            f"Amostra Falhada {i+1}", step=10)
    col1.write(f"Amostras Falhadas:")

    for j in range(num_falhas):
        col1.write(f"Amostra {j+1}: {amostras_falhadas[j]}")
        if xmin == 0:
            xmin = amostras_falhadas[j]
        if amostras_falhadas[j] < xmin:
            xmin = amostras_falhadas[j]
        if xmax == 0:
            xmax = amostras_falhadas[j]
        if amostras_falhadas[j] > xmax:
            xmax = amostras_falhadas[j]

if num_censuradas > 0:
    st.sidebar.write("Amostras Censuradas:")
    for i in range(num_censuradas):
        amostras_censuradas[i] = st.sidebar.number_input(
            f"Amostra Censurada {i+num_falhas+1}", step=10)
    col2.write(f"Amostras Censuradas:")
    for j in range(num_censuradas):
        col2.write(f"Amostra {j+num_falhas+1}: {amostras_censuradas[j]}")
        if xmin == 0:
            xmin = amostras_falhadas[j]
        if amostras_falhadas[j] < xmin:
            xmin = amostras_falhadas[j]
        if xmax == 0:
            xmax = amostras_falhadas[j]
        if amostras_falhadas[j] > xmax:
            xmax = amostras_falhadas[j]

if xmin > 0 and xmax > 0:
    xlim_min = st.sidebar.slider(
        "Mínimo X Gráfico Weibull:", xmin//3, xmin, xmin//2)
    xlim_max = st.sidebar.slider(
        "Máximo X Gráfico Weibull:", xmax, xmax*3, xmax*2)

CI = st.sidebar.slider("Intervalo de Confiança", 0.50, 0.99, 0.90)

method = st.sidebar.selectbox("Escolha o Método",
                              ("MLE", "LS", "RRX",
                                  "RRY"),
                              help="""‘MLE’ (Estimativa de Máxima Verossimilhança),
                              ‘LS’ (Mínimos Quadrados),
                              ‘RRX’ (Rank Regressão em X),
                              ‘RRY’ (Rank Regressão em Y).
                              LS irá testar RRX e RRYe retornar o melhor."""
                              )
if method == "MLE":
    optimizer = st.sidebar.selectbox("Escolha o Otimizador",
                                     ("Best", "TNC", "L-BFGS-B",
                                      "Nelder-Mead", "Powell"),
                                     help="""Habilitado apenas para o método MLE. 
                                     Escolha entre 'Best', para testar todas e aplicar a melhor opção,
                                     'TNC' (Newton Truncado),
                                     'L-BFGS-B' (Broyden–Fletcher–Goldfarb–Shanno de Memória Limitada),
                                     'Nelder-Mead' ou 'Powell"""
                                     )
else:
    optimizer = None
    st.sidebar.selectbox("Escolha o Otimizador", ("None"), disabled=True,
                         help="""Habilitado apenas para o método MLE. 
                                     Escolha entre 'Best', para testar todas e aplicar a melhor opção,
                                     'TNC' (Newton Truncado),
                                     'L-BFGS-B' (Broyden–Fletcher–Goldfarb–Shanno de Memória Limitada),
                                     'Nelder-Mead' ou 'Powell"""
                         )


def calculo_weibull(amostras_falhadas, amostras_censuradas, CI, optimizer, method, quantiles):
    failures = []
    censored = []

    for i in amostras_falhadas:
        failures.append(amostras_falhadas[i])

    for j in amostras_censuradas:
        censored.append(amostras_censuradas[j])
    fig = plt.figure()
    plt.gcf().set_dpi(60)
    plt.subplot(121)
    fit = Fit_Weibull_2P(failures=failures,
                         right_censored=censored,
                         CI=CI,
                         optimizer=optimizer,
                         method=method,
                         quantiles=True
                         )
    dist_1 = Weibull_Distribution(alpha=fit.alpha, beta=fit.beta)
    dist_1.PDF(label=dist_1.param_title_long)
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0.01, 0.99)
    plt.subplot(122)
    dist_1.PDF(label=dist_1.param_title_long)

    st.subheader(f"Resultados de Fit Weibull 2P({CI*100}% CI):")
    f"Otimizador: {fit.optimizer}"
    f"Método: {fit.method}"
    f"Quantidade Amostras Falhadas = {len(failures)}"
    f"Quantidade Amostras Censuradas = {len(censored)} "
    fit.results
    fit.goodness_of_fit
    fit.goodness_of_fit
    fit.quantiles

    st.pyplot(fig)

    return

if num_falhas + num_censuradas > 3:
    calcular_Button = st.sidebar.button("Calcular", disabled=False, help="Número mínimo de amostras = 4 (ideal mínimo 6)")
else:
    calcular_Button = st.sidebar.button(
        "Calcular", disabled=True, help="Número mínimo de amostras = 4 (ideal mínimo 6)")
if calcular_Button:
    calculo_weibull(amostras_falhadas, amostras_censuradas,
                    CI, optimizer, method, quantiles)
