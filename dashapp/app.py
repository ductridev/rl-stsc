import matplotlib
matplotlib.use("Agg")  # avoid Tkinter GUI backend
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from test_traffic import (
    test_base_simulation,
    test_actuated_simulation,
    test_dqn_simulation,
)
from src.utils import import_test_configuration, set_test_path

# -------------------- Metrics --------------------
METRICS = [
    "density_avg",
    "travel_time_avg",
    "outflow_avg",
    "queue_length_avg",
    "waiting_time_avg",
    "travel_delay_avg",
]

# -------------------- Color Scheme --------------------
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#42B883',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'light': '#F8F9FA',
    'dark': '#2C3E50',
    'gray': '#6C757D',
    'white': '#FFFFFF'
}

# -------------------- Run simulations --------------------
def run_all(config_path, sims=("base", "actuated", "skrl_dqn")):
    config = import_test_configuration(config_path)
    path = set_test_path(config["models_path_name"])

    results = {}
    for sim in sims:
        if sim == "base":
            result = test_base_simulation(config, path)
        elif sim == "actuated":
            result = test_actuated_simulation(config, path)
        elif sim == "skrl_dqn":
            result = test_dqn_simulation(config, path)
        else:
            continue

        if result is None:
            results[sim] = {"completion": {}, "metrics": {}}
        else:
            comp, metrics = result
            results[sim] = {"completion": comp, "metrics": metrics}
    return results, config, path

def build_comparison_plots(path, episode=0, metrics=None, names=None):
    if metrics is None:
        metrics = METRICS
    if names is None:
        names = ["base", "actuated", "skrl_dqn"]

    # Color mapping for different simulations
    color_map = {
        "base": COLORS['primary'],
        "actuated": COLORS['accent'],
        "skrl_dqn": COLORS['success']
    }

    figures = []

    for metric in metrics:
        data = {}
        for name in names:
            filename = os.path.join(path, f"{name}_{metric}_episode_{episode}.txt")
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data[name] = [float(line.strip()) for line in f if line.strip()]

        fig = go.Figure()
        for name in names:
            if name in data:
                fig.add_trace(go.Scatter(
                    y=data[name],
                    x=list(range(len(data[name]))),
                    mode="lines+markers",
                    name=name.upper(),
                    line=dict(color=color_map.get(name, COLORS['gray']), width=3),
                    marker=dict(size=6, color=color_map.get(name, COLORS['gray']))
                ))

        fig.update_layout(
            title={
                'text': f"{metric.replace('_', ' ').title()} Comparison",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': COLORS['dark'], 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Time Intervals (5 Minutes)",
            yaxis_title=metric.replace("_", " ").title(),
            xaxis=dict(
                title=dict(font=dict(size=14, color=COLORS['dark'])),
                tickfont=dict(size=12, color=COLORS['gray']),
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title=dict(font=dict(size=14, color=COLORS['dark'])),
                tickfont=dict(size=12, color=COLORS['gray']),
                gridcolor='rgba(0,0,0,0.1)',
                showgrid=True
            ),
            legend=dict(
                x=1.02, y=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS['gray'],
                borderwidth=1,
                font=dict(size=12)
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=60, t=60, b=60)
        )
        figures.append(fig)

    return figures

# -------------------- Dash App Setup --------------------
app = dash.Dash(__name__)
app.title = "Traffic Signal Control Dashboard"

# Custom CSS styling
app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.H1(
                "Traffic Signal Control Dashboard",
                style={
                    'color': COLORS['white'],
                    'margin': '0',
                    'fontSize': '2.5rem',
                    'fontWeight': '300',
                    'letterSpacing': '1px'
                }
            ),
            html.P(
                "Real-time traffic simulation analysis and comparison",
                style={
                    'color': COLORS['white'],
                    'opacity': '0.9',
                    'margin': '10px 0 0 0',
                    'fontSize': '1.1rem'
                }
            )
        ], style={
            'textAlign': 'center',
            'padding': '40px 20px'
        })
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]})',
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    }),

    # Main Content Container
    html.Div([
        # Control Panel
        html.Div([
            html.Div([
                html.H3("Configuration", style={
                    'color': COLORS['dark'],
                    'marginBottom': '20px',
                    'fontSize': '1.5rem',
                    'fontWeight': '600'
                }),
                html.Div([
                    html.Label("Config File Path:", style={
                        'color': COLORS['dark'],
                        'fontWeight': '500',
                        'marginBottom': '8px',
                        'display': 'block'
                    }),
                    dcc.Dropdown(
                        id="config-path",
                        options=[
                            {"label": "testing_testngatu6x1-1", "value": "config/testing_testngatu6x1-1.yaml"},
                            {"label": "testing_testngatu6x1-2", "value": "config/testing_testngatu6x1-2.yaml"},
                            {"label": "testing_testngatu6x1-3", "value": "config/testing_testngatu6x1-3.yaml"},
                            {"label": "testing_testngatu6x1-4", "value": "config/testing_testngatu6x1-4.yaml"},
                            {"label": "testing_testngatu6x1-5", "value": "config/testing_testngatu6x1-5.yaml"},
                            {"label": "testing_testngatu6x1-6", "value": "config/testing_testngatu6x1-6.yaml"},
                        ],
                        value="config/testing_testngatu6x1-4.yaml",
                        style={
                            'width': '100%',
                            'padding': '12px',
                            'border': f'2px solid {COLORS["light"]}',
                            'borderRadius': '8px',
                            'fontSize': '14px',
                            'marginBottom': '15px',
                            'transition': 'border-color 0.3s ease',
                            'outline': 'none'
                        }
                    ),
                    html.Button(
                        "Run Simulations",
                        id="run-btn",
                        n_clicks=0,
                        style={
                            'backgroundColor': COLORS['primary'],
                            'color': COLORS['white'],
                            'border': 'none',
                            'padding': '12px 30px',
                            'borderRadius': '8px',
                            'fontSize': '16px',
                            'fontWeight': '600',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        }
                    )
                ])
            ], style={
                'backgroundColor': COLORS['white'],
                'padding': '25px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.07)',
                'border': f'1px solid {COLORS["light"]}'
            })
        ], style={'marginBottom': '30px'}),

        # Configuration Info
        html.Div(id="config-info", style={'marginBottom': '30px'}),

        # Plots Section
        html.Div([
            html.H3("Comparative Analysis", style={
                'color': COLORS['dark'],
                'marginBottom': '20px',
                'fontSize': '1.8rem',
                'fontWeight': '600'
            }),
            html.Div(id="plots-container", children=[])
        ])
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px'
    })
], style={
    'fontFamily': 'Arial, Helvetica, sans-serif',
    'backgroundColor': '#F8F9FA',
    'minHeight': '100vh',
    'margin': '0',
    'padding': '0'
})

# -------------------- Callbacks --------------------
@app.callback(
    [Output("config-info", "children"),
     Output("plots-container", "children")],
    Input("run-btn", "n_clicks"),
    State("config-path", "value")
)
def run_and_update(n_clicks, config_path):
    if n_clicks == 0:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle", style={
                    'color': COLORS['primary'],
                    'fontSize': '24px',
                    'marginRight': '10px'
                }),
                html.Span("Enter configuration path and click 'Run Simulations' to begin analysis.", style={
                    'color': COLORS['dark'],
                    'fontSize': '16px',
                    'fontWeight': '500'
                })
            ], style={
                'backgroundColor': COLORS['white'],
                'padding': '20px',
                'borderRadius': '12px',
                'border': f'2px solid {COLORS["primary"]}',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'textAlign': 'center'
            })
        ]), []

    try:
        results, config, path = run_all(config_path)

        # Configuration info with enhanced styling
        config_info = html.Div([
            html.Div([
                html.H4("Configuration Details", style={
                    'color': COLORS['dark'],
                    'marginBottom': '15px',
                    'fontSize': '1.3rem',
                    'fontWeight': '600'
                }),
                html.Div([
                    html.Div([
                        html.Strong("File Path: ", style={'color': COLORS['gray']}),
                        html.Span(config_path, style={'color': COLORS['dark']})
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Strong("Max Steps: ", style={'color': COLORS['gray']}),
                        html.Span(str(config.get('max_steps', 'N/A')), style={'color': COLORS['dark']})
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Strong("Traffic Lights: ", style={'color': COLORS['gray']}),
                        html.Span(str(len(config.get('traffic_lights', []))), style={'color': COLORS['dark']})
                    ])
                ])
            ], style={
                'backgroundColor': COLORS['white'],
                'padding': '25px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.07)',
                'border': f'1px solid {COLORS["light"]}'
            })
        ])

        # Generate plots with enhanced styling
        figures = build_comparison_plots(path, episode=1, metrics=METRICS, names=list(results.keys()))
        plots = [
            html.Div([
                dcc.Graph(
                    figure=fig,
                    style={'height': '400px'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                    }
                )
            ], style={
                'backgroundColor': COLORS['white'],
                'padding': '20px',
                'borderRadius': '12px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.07)',
                'border': f'1px solid {COLORS["light"]}',
                'marginBottom': '20px'
            })
            for fig in figures
        ]

        return config_info, plots

    except Exception as e:
        error_info = html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={
                    'color': COLORS['danger'],
                    'fontSize': '24px',
                    'marginRight': '10px'
                }),
                html.Span(f"Error running simulations: {str(e)}", style={
                    'color': COLORS['danger'],
                    'fontSize': '16px',
                    'fontWeight': '500'
                })
            ], style={
                'backgroundColor': COLORS['white'],
                'padding': '20px',
                'borderRadius': '12px',
                'border': f'2px solid {COLORS["danger"]}',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'textAlign': 'center'
            })
        ])
        return error_info, []

if __name__ == "__main__":
    app.run(debug=True)