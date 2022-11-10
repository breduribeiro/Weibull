from statistics import quantiles
import streamlit as st
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points
import matplotlib.pyplot as plt
import pandas as pd

# Programa para análise estatística de vida de amostras através do cálculo Weibull
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
    \nOpcional: Defina o equivalente a 1 vida. Neste caso será inserida indicação de chance de falha para 1 vida no gráfico.
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

# Indicação do número de amostras falhadas e censuradas e vida de cada amostra
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

# Definição dos limites mínimos e máximos de X (vida) do gráfico Weibull
if xmin > 0 and xmax > 0:
    xlim_min = st.sidebar.slider(
        "Mínimo X Gráfico Weibull:", xmin//3, xmin, xmin//2)
    xlim_max = st.sidebar.slider(
        "Máximo X Gráfico Weibull:", xmax, xmax*3, xmax*2)

# Definição do Intervalo de Confiança (C)
CI = st.sidebar.slider("Intervalo de Confiança - C", 0.50, 0.99, 0.90)

# Definição da Confiabilidade (B para Probabilidade de falha, R para Probabilidade de Sucesso)
B = st.sidebar.selectbox("Defina o B desejado (opcional)",
                         ('Nenhum', 'B5', 'B10', 'B25', 'B50'),
                         help="""O B define a Probalidade (R) desejada.
                        A seleção escolhida irá indicar a vida prevista 
                        para o B escolhido.""")
if B == 'Nenhum':
    B = None
else:
    B = float(B[1:])

# Definição da estimativa de 1 vida (em qualquer unidade temporal) para a população
vida = st.sidebar.number_input(
    "Defina o equivalente a 1 vida (opicional)", step=10)

# Definição do método utilizado (MLE - Verossimilhança ou LS (RRX, RRY) - Mínimos Quadrados)
# Apenas o método MLE permite seleção do Otimizador ("TNC", "L-BFGS-B", "Nelder-Mead" ou "Powell")
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

# Função para o cálculo Weibull


def calculo_weibull(amostras_falhadas, amostras_censuradas, CI, optimizer, method, B):
    failures = []
    censored = []
    for i in amostras_falhadas:
        failures.append(amostras_falhadas[i])

    for j in amostras_censuradas:
        censored.append(amostras_censuradas[j])

# Criação do gráfico Weibull
    try:
        fig1, ax1 = plt.subplots()
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

        # Resultados do cálculo Weibull
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

        Abaixo = fit.quantiles.loc[fit.quantiles['Lower Estimate'] < vida]
        Acima = fit.quantiles.loc[fit.quantiles['Lower Estimate'] > vida]

# Anotação para vida equivalente ao B escolhido
        if B != None:
            R = B/100
            Vida_B = fit.distribution.CDF(
                CI_type='time', CI_y=R, show_plot=False)
            B_annotate = f'{Vida_B[0]:.0f}'
            ax1.annotate(
                f"""B{B:.0f}: {B_annotate}""",
                xy=(Vida_B[0], R),
                xytext=(-20, 5),
                textcoords='offset points'
            )
            plt.plot([xlim_min, Vida_B[0], Vida_B[0]],
                     [R, R, 0.01], color='blue')

# Anotação para percentil estimado para 1 vida
        B_Vida = 0
        y = 0.5
        if vida > 0:
            while round(B_Vida) != vida:
                B_Vida = fit.distribution.CDF(
                    CI_type='time', CI_y=y, show_plot=False)
                y = float(y*vida/B_Vida[0])
                B_Vida = B_Vida[0]
                if y < 0.01 or y > 0.99:
                    break
            if y > 0.01 and y < 0.99:
                texto_annotate = f'{y:.1%}'
                texto_annotate = texto_annotate.replace('.', ',')
                ax1.annotate(
                    f"""1 vida: {texto_annotate}""",
                    xy=(xlim_min, y),
                    xytext=(5, 8),
                    textcoords='offset points'
                )
                plt.plot([xlim_min, vida, vida], [
                    y, y, 0.01], color='blue', linestyle='dashed')
        st.pyplot(fig1)

# Cálculo da distribuição PDF
        fig2, ax2 = plt.subplots()
        dist_1 = Weibull_Distribution(alpha=fit.alpha, beta=fit.beta)
        yvalues = dist_1.PDF()
        plt.title(f"""Distribuição Weibull
                    \n(α={alpha}; β={beta})""")
        plt.xlabel('Vida')
        plt.ylabel('Densidade')

        st.pyplot(fig2)

    except ValueError as erro:
        st.error(erro)
        # "Não foi possível realizar a regressão com o Método selecionado. Tente novamente, escolhendo outro Método.")
    return


# Geração do Botão para realização do cálculo
if num_falhas >= 5:
    calcular_Button = st.sidebar.button(
        "Calcular", disabled=False, help="Número mínimo de amostras falhadas = 5 (ideal mínimo 6)")

else:
    calcular_Button = st.sidebar.button(
        "Calcular", disabled=True, help="Número mínimo de amostras falhadas = 5 (ideal mínimo 6)")
if calcular_Button:
    calculo_weibull(amostras_falhadas, amostras_censuradas,
                    CI, optimizer, method, B)

string_Linkedin = """<a href='https://www.linkedin.com/in/breno-ribeiro-8b062890/'>
                     <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2RYeN56EvozwyyxYGDw4dTu-pbUZyNxnF93zSLUcOlQ&s'
                     alt='Linkedin' style='width:20px;height:20px;'>
                     </a>"""
string_Email = """<a href='mailto:breduribeiro@gmail.com'>
                  <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAAF43Dcua1axzMUg1OIG-xjuKerm29tGX7SZnbskgAw&s'
                  alt='Mail to' style='width:20px;height:20px;'>
                  </a>"""
st.sidebar.markdown(
    f"""Desenvolvido por **Breno Ribeiro** {string_Linkedin} {string_Email}""", unsafe_allow_html=True)
st.sidebar.write("")
