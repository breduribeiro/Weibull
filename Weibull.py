from ast import Lambda
from operator import truediv
from statistics import quantiles
import streamlit as st
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

st.title(
    "[Weibull](https://pt.wikipedia.org/wiki/Distribui%C3%A7%C3%A3o_de_Weibull)")

expand = st.expander("Instruções", expanded=False)
with expand:
    """Os campos de preenchimento para realização do cálculo Weibull estão na barra
    à esquerda. Caso não a esteja visualizando, é necessário expandi-la na seta no canto superior esquerdo.
    \nSelecione o número de amostras que falharam (Amostras Falhadas). O cálculo Weibull exige o mínimo de
    4 amostras, porém é recomendado no mínimo 6 amostras.
    \nSelecione o número de amostras que foram interrompidas (Amostas Censuradas).
    \nPara cada amostra falhada ou censurada, insira, no campo indicado, a sua vida.
    \nSelecione o Intervalo de Confiança CI (entre 50% e 99%).
    \nEscolha o método de linearização:
    \n[Mínimos Quadrados (RRX, RRY ou LS)](https://pt.wikipedia.org/wiki/M%C3%A9todo_dos_m%C3%ADnimos_quadrados)
    (LS o software irá escolher a melhor opção entre RRX e RRY)
    \n[Máxima Verossimilhança (MLE)](https://pt.wikipedia.org/wiki/M%C3%A1xima_verossimilhan%C3%A7a)
    \nEscolha o método de otimização (apenas disponível para o método MLE) ou 'Best' para testar todas e 
    escolher a melhor opção:
    \n[Newton Truncado (TNC)](https://en.wikipedia.org/wiki/Truncated_Newton_method)
    \n[L-BFGS-B' (Broyden–Fletcher–Goldfarb–Shanno de Memória Limitada)](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
    \n[Nelder-Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
    \n[Powell](https://en.wikipedia.org/wiki/Powell%27s_method)
    \nClique em 'Calcular'
    \nSerão apresentados as vidas das amostras falhadas e censuradas, o Método e o Otimizador escolhidos,
    os parâmetros Alfa e Beta, os parâmetros para o cálculo da regressão linear, a tabela dos percentis de falha
    e os gráficos Weibull de Probabilidade e Distribuição. Todas as tabelas e gráficos podem ser expandidos
    (ícone de expansão aparece no canto superior direito do elemento selecionado), selecionados e copiados
    para a área de transferência."""

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
    fmin = min(amostras_falhadas.values())
    fmax = max(amostras_falhadas.values())
    xmin = fmin
    xmax = fmax
if num_censuradas > 0:
    st.sidebar.write("Amostras Censuradas:")
    for i in range(num_censuradas):
        amostras_censuradas[i] = st.sidebar.number_input(
            f"Amostra Censurada {i+num_falhas+1}", step=10)
    col2.write(f"Amostras Censuradas:")
    for j in range(num_censuradas):
        col2.write(f"Amostra {j+num_falhas+1}: {amostras_censuradas[j]}")
    cmin = min(amostras_censuradas.values())
    cmax = min(amostras_censuradas.values())
    if num_falhas > 0:
        xmin = min(fmin, cmin)
        xmax = max(fmax, cmax)

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

    fig = plt.figure(figsize=(12, 5))
    fit = Fit_Weibull_2P(failures=failures,
                         right_censored=censored,
                         CI=CI,
                         optimizer=optimizer,
                         method=method,
                         quantiles=True,
                         )

    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0.01, 0.99)
    st.subheader(f"Resultados de Fit Weibull 2P({CI*100}% CI):")
    f"Otimizador: {fit.optimizer}"
    f"Método: {fit.method}"
    f"Quantidade Amostras Falhadas = {len(failures)}"
    f"Quantidade Amostras Censuradas = {len(censored)} "
    fit.results
    fit.goodness_of_fit
    fit.quantiles
    cont1 = st.container()
    cont2 = st.container()
    with cont1:
        st.pyplot(fig)
    with cont2:
        fig = plt.figure(figsize=(12, 8))
        dist_1 = Weibull_Distribution(alpha=fit.alpha, beta=fit.beta)
        dist_1.PDF(label="dist_1.param_title_long")
        st.pyplot(fig)
    return

if num_falhas  >= 4:
    calcular_Button = st.sidebar.button(
        "Calcular", disabled=False, help="Número mínimo de amostras falhadas = 4 (ideal mínimo 6)")
    st.sidebar.write("")
    st.sidebar.write("")
else:
    calcular_Button = st.sidebar.button(
        "Calcular", disabled=True, help="Número mínimo de amostras falhadas = 4 (ideal mínimo 6)")
if calcular_Button:
    calculo_weibull(amostras_falhadas, amostras_censuradas,
                    CI, optimizer, method, quantiles)
