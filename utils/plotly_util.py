from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go

#setting offilne
plotly.offline.init_notebook_mode(connected=True)


def draw_scatter(x, y):
    pass

if __name__ == '__main__':
    N = 100
    random_x = np.linspace(0,1,N)
    random_y0 = np.random.randn(N)+5
    random_y1 = np.random.randn(N)
    random_y2 = np.random.randn(N)-5

    #Create traces
    trace0 = go.Scatter(
        x = random_x,
        y = random_y0,
        mode = 'markers',
        name = 'markers'
    )
    trace1 = go.Scatter(
        x = random_x,
        y = random_y1,
        mode = 'lines+markers',
        name = 'lines+markers'
    )
    trace2 = go.Scatter(
        x = random_x,
        y = random_y2,
        mode = 'lines',
        name = 'lines'
    )
    data = [trace0,trace1,trace2]
    py.iplot(data)