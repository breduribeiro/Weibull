from statistics import quantiles
import streamlit as st
import streamlit.components.v1 as components
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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
    \n[Mínimos Quadrados (RRX, RRY ou LS)](https://reliability.readthedocs.io/en/latest/How%20does%20Least%20Squares%20Estimation%20work.html)
    (LS o software irá escolher a melhor opção entre RRX e RRY)
    \n[Máxima Verossimilhança (MLE)](https://reliability.readthedocs.io/en/latest/How%20does%20Maximum%20Likelihood%20Estimation%20work.html)
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

CI = st.sidebar.slider("Intervalo de Confiança - C", 0.50, 0.99, 0.90)

vida = st.sidebar.number_input(
    "Defina o equivalente a 1 vida (opicional)", step=10)

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

    fig, ax = plt.subplots()
    try:
        fit = Fit_Weibull_2P(failures=failures,
                             right_censored=censored,
                             CI=CI,
                             optimizer=optimizer,
                             method=method,
                             quantiles=True,
                             )

        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0.01, 0.99)
        alpha = f'{fit.alpha:_.2f}'
        alpha = alpha.replace('.', ',').replace('_', '.')
        beta = f'{fit.beta:_.2f}'
        beta = beta.replace('.', ',').replace('_', '.')
        plt.title(f"""Probabilidade Weibull ({CI:.0%} CI)
                    \n(α={alpha}; β={beta})""")
        plt.xlabel('Vida')
        plt.ylabel('Probabilidade de Falha')
        plt.legend().remove()
        st.subheader(f"Resultados de Fit Weibull 2P({CI:.0%} CI):")
        f"Otimizador: {fit.optimizer}"
        f"Método: {fit.method}"
        f"Quantidade Amostras Falhadas = {len(failures)}"
        f"Quantidade Amostras Censuradas = {len(censored)} "

        st.dataframe(pd.DataFrame(fit.results).style.format(
            decimal=',', thousands='.', precision=2))
        st.dataframe(pd.DataFrame(fit.goodness_of_fit).style.format(
            decimal=',', thousands='.', precision=2))
        st.dataframe(pd.DataFrame(fit.quantiles).style.format(
            decimal=',', thousands='.', precision=2))

        cont1 = st.container()
        cont2 = st.container()

        with cont1:
            Abaixo = fit.quantiles.loc[fit.quantiles['Lower Estimate'] < vida]
            Acima = fit.quantiles.loc[fit.quantiles['Lower Estimate'] > vida]
            if Abaixo.size > 0 and Acima.size > 0:
                x = [
                    np.log10(
                        Abaixo.values[(Abaixo['Lower Estimate'].size)-1][1]),
                    np.log10(Acima.values[0][1])]
                y = [
                    np.log10(
                        Abaixo.values[(Abaixo['Lower Estimate'].size)-1][0]),
                    np.log10(Acima.values[0][0])]
                interpolate = interp1d(x, y)
                Falha_Vida = 10**(interpolate(np.log10(vida)))
                texto_annotate = f'{Falha_Vida:.1%}'
                texto_annotate = texto_annotate.replace('.', ',')
                ax.annotate(
                    f"""Probabilidade máxima de \nfalha para 1 vida: {texto_annotate}""",
                    xy=(vida, Falha_Vida),
                    xytext=(-100, 40),
                    textcoords='offset points',
                    bbox={'boxstyle': 'round', 'fc': 'w'},
                    arrowprops={'arrowstyle': '->'}
                )

            st.pyplot(fig)

        with cont2:
            fig, ax = plt.subplots()
            dist_1 = Weibull_Distribution(alpha=fit.alpha, beta=fit.beta)
            yvalues = dist_1.PDF()
            plt.title(f"""Distribuição Weibull
                        \n(α={alpha}; β={beta})""")
            plt.xlabel('Vida')
            plt.ylabel('Densidade')

            for axes in fig.axes:
                for line in axes.get_lines():
                    # get the x and y coords
                    xy_data = line.get_xydata()
                    df = pd.DataFrame(xy_data)
                    mean = (df.loc[df[1].idxmax()])
                    ax.annotate(
                        f'Mediana = {mean[0]:.0f}',
                        xy=(mean[0], mean[1]),
                        xytext=(15, 15),
                        textcoords='offset points',
                        bbox={'boxstyle': 'round', 'fc': 'w'},
                        arrowprops={'arrowstyle': '->'}
                    )
            st.pyplot(fig)

    except ValueError as erro:
        st.error(
            "Não foi possível realizar a regressão com o Método selecionado. Tente novamente, escolhendo outro Método.")
    return

if num_falhas >= 4:
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
