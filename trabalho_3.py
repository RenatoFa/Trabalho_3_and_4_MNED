import numpy as np
import plotly.graph_objects as go

# Parâmetros físicos
ALPHA_VALUES = [0.01, 0.1, 0.5]
K_VALUES = [0.02, 0.1, 0.5]
LX = 1.0
CE = 1.0
T_FINAL = 1.0

# Parâmetros numéricos
NX_VALUES = [10, 50, 100, 500]
NT = 1000  # Número de passos no tempo
DT = T_FINAL / NT

# Valor de nx para a solução de referência
NX_REF = 1000


def construct_matrix(alpha, k, dx, dt, nx):
    """
    Constrói a matriz do sistema A para o método implícito.
    """
    A = np.zeros((nx, nx))
    coeff = alpha / dx**2
    main_diag = (1 / dt) + 2 * coeff + k
    off_diag = -coeff

    # Preenchimento da matriz
    np.fill_diagonal(A[1:-1, 1:-1], main_diag)
    np.fill_diagonal(A[1:-1, :-2], off_diag)
    np.fill_diagonal(A[1:-1, 2:], off_diag)

    # Condição de contorno em x=0 (Dirichlet)
    A[0, 0] = 1
    # Condição de Neumann em x=Lx
    A[-1, -2] = -1 / (2 * dx)
    A[-1, -1] = 1 / (2 * dx)
    return A


def compute_solution(alpha, k, nx, nt, dt, lx, ce):
    """
    Calcula a solução numérica para os parâmetros fornecidos.
    """
    dx = lx / (nx - 1)
    x = np.linspace(0, lx, nx)
    C = np.zeros(nx)
    C[0] = ce
    A = construct_matrix(alpha, k, dx, dt, nx)

    for _ in range(nt):
        b = C / dt
        b[0] = ce  # Condição em x=0
        b[-1] = 0  # Condição de Neumann em x=Lx
        C = np.linalg.solve(A, b)
    return x, C


def add_trace_to_figure(fig, x, C, name, legendgroup, line_style):
    """
    Adiciona uma curva ao gráfico interativo.
    """
    trace = go.Scatter(
        x=x,
        y=C,
        mode='lines',
        line=line_style,
        name=name,
        legendgroup=legendgroup,
        hovertemplate='x: %{x}<br>C: %{y}<extra></extra>',
        opacity=1
    )
    fig.add_trace(trace)


def main():
    # Criação da figura interativa
    fig = go.Figure()

    # Loop sobre os parâmetros alpha e k
    for alpha in ALPHA_VALUES:
        for k in K_VALUES:
            # Cálculo da solução de referência
            x_ref, C_ref = compute_solution(alpha, k, NX_REF, NT, DT, LX, CE)
            name_ref = f'Sol. ref (α={alpha}, k={k})'
            legendgroup = f'alpha_{alpha}_k_{k}'
            line_style_ref = dict(dash='dash', color='black', width=2)
            add_trace_to_figure(fig, x_ref, C_ref, name_ref,
                                legendgroup, line_style_ref)

            # Loop sobre os valores de nx
            for nx in NX_VALUES:
                x, C = compute_solution(alpha, k, nx, NT, DT, LX, CE)
                name = f'α={alpha}, k={k}, nx={nx}'
                line_style = dict(width=2)
                add_trace_to_figure(fig, x, C, name, legendgroup, line_style)

    # Ajustando o layout e a legenda
    fig.update_layout(
        title='Distribuição de C após Tempo Final',
        xaxis_title='Posição x',
        yaxis_title='Concentração C',
        legend_title='Curvas',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            itemclick='toggle',
            itemdoubleclick='toggleothers',
            bgcolor='LightSteelBlue',
            bordercolor='Black',
            borderwidth=1
        )
    )

    # Converter a figura em HTML
    html_str = fig.to_html(full_html=True)

    # Código JavaScript para manipular o evento de clique na legenda
    js_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var myPlot = document.getElementsByClassName('js-plotly-plot')[0];
        var highlightedTraces = [];

        myPlot.on('plotly_legendclick', function(data) {
            var clickedTrace = data.curveNumber;
            var traceIndex = highlightedTraces.indexOf(clickedTrace);

            if (traceIndex === -1) {
                // Não está destacado, adicionar ao array
                highlightedTraces.push(clickedTrace);
            } else {
                // Já está destacado, remover do array
                highlightedTraces.splice(traceIndex, 1);
            }

            var updateTraces = [];
            var opacity = [];
            var lineWidth = [];

            if (highlightedTraces.length === 0) {
                // Nenhuma linha destacada, resetar para o estado original
                for (var i = 0; i < myPlot.data.length; i++) {
                    opacity.push(1);
                    lineWidth.push(2);
                    updateTraces.push(i);
                }
            } else {
                // Atualizar as propriedades das linhas
                for (var i = 0; i < myPlot.data.length; i++) {
                    if (highlightedTraces.indexOf(i) !== -1) {
                        // Linha destacada
                        opacity.push(1);
                        lineWidth.push(4);
                    } else {
                        // Linha ofuscada
                        opacity.push(0.2);
                        lineWidth.push(1);
                    }
                    updateTraces.push(i);
                }
            }

            Plotly.restyle(myPlot, {'opacity': opacity, 'line.width': lineWidth}, updateTraces);

            // Impedir o comportamento padrão
            return false;
        });
    });
    </script>
    """

    # Inserir o código JavaScript antes da tag de fechamento </body>
    html_str = html_str.replace('</body>', js_code + '</body>')

    # Salvar o HTML modificado em um arquivo
    with open('grafico_interativo.html', 'w') as f:
        f.write(html_str)


if __name__ == "__main__":
    main()
