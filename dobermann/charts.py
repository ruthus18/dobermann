import altair as alt
import pandas as pd
import math


def get_extended_equity_chart(df: pd.DataFrame) -> alt.VConcatChart:
    """Get equity chart extended with leverage and drawdown info

    Required columns for `df`:
        * Time: datetime
        * Equity: float
        * Drawdown: float
        * Leverage Used: float
    """
    hover_selection = get_hover_selection()

    equity_chart = get_equity_chart(df, hover_selection)
    leverage_chart = get_leverage_chart(df, hover_selection)
    drawdown_chart = get_drawdown_chart(df, hover_selection)

    return alt.vconcat(equity_chart, leverage_chart, drawdown_chart)


def get_hover_selection() -> alt.Selection:
    return alt.selection(type='single', nearest=True, on='mouseover', fields=['Time'], empty='none')


def get_equity_chart(df: pd.DataFrame, hover_selection: alt.Selection | None = None) -> alt.LayerChart:
    """Get equity chart with smart hover tooltip

    Required columns for `df`:
        * Time: datetime
        * Equity: float
    """
    if not hover_selection:
        hover_selection = get_hover_selection()

    equity_min_y_scale = (math.floor(df['Equity'].min() / 100) - 0.5) * 100
    equity_max_y_scale = (math.ceil(df['Equity'].max() / 100) + 0.5) * 100

    equity_line = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(
                'Time',
                axis=alt.Axis(grid=False, title='')
            ),
            y=alt.Y(
                'Equity',
                axis=alt.Axis(title=''),
                scale=alt.Scale(domain=[equity_min_y_scale, equity_max_y_scale])
            ),
        )
        .transform_filter(filter={'field': 'Equity', 'valid': True})
    )
    equity_start_line = (
        alt.Chart(pd.DataFrame({'y': [1000]}))
        .mark_rule(size=0.2)
        .encode(y='y', color=alt.value('red'))
    )
    equity_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='Time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('Time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('Equity', format='.2f')
        ])
        .add_selection(hover_selection)
    )
    equity_hover_points = (
        equity_line
        .mark_point(filled=True)
        .encode(opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)))
    )
    equity_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='Time')
        .transform_filter(hover_selection)
    )

    return alt.layer(
        equity_line, equity_start_line, equity_hover_selectors, equity_hover_rule, equity_hover_points
    ).properties(
        title='Equity', width=1000, height=300
    )


def get_leverage_chart(df: pd.DataFrame, hover_selection: alt.Selection | None = None) -> alt.LayerChart:
    """Get used leverage chart with smart hover tooltip

    Required columns for `df`:
        * Time: datetime
        * Leverage_used: float
    """
    if not hover_selection:
        hover_selection = get_hover_selection()
        
    leverage_min_y_scale = 0
    leverage_max_y_scale = (math.ceil(df['Leverage Used'].max()) + 0.5)

    leverage_hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                'Time',
                axis=alt.Axis(grid=False, title='')
            ),
            y=alt.Y(
                'Leverage Used',
                axis=alt.Axis(title=''),
                scale=alt.Scale(domain=[leverage_min_y_scale, leverage_max_y_scale])
            ),
            color=alt.value('darkgreen'),
            opacity=alt.condition(hover_selection, alt.value(1), alt.value(0.5))
        )
    )
    leverage_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='Time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('Time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('Leverage Used', format='.2f')
        ])
        .add_selection(hover_selection)
    )
    leverage_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='Time')
        .transform_filter(hover_selection)
    )

    return alt.layer(
        leverage_hist, leverage_hover_selectors, leverage_hover_rule
    ).properties(
        title='Leverage Used', width=1000, height=150
    )


def get_drawdown_chart(df: pd.DataFrame, hover_selection: alt.Selection | None = None) -> alt.LayerChart:
    """Get drawdown chart with smart hover tooltip

    Required columns for `df`:
        * Time: datetime
        * Drawdown: float
    """
    if not hover_selection:
        hover_selection = get_hover_selection()
        
    drawdown_line = (
        alt.Chart(df)
        .mark_line(color='red')
        .encode(
            x=alt.X('Time', axis=alt.Axis(grid=False, title='')),
            y=alt.Y('Drawdown', axis=alt.Axis(title='')),
        )
        .transform_filter(filter={'field': 'Drawdown', 'valid': True})
    )
    drawdown_hover_selectors = (
        alt.Chart(df)
        .mark_rule(size=10)
        .encode(x='Time', opacity=alt.value(0), tooltip=[
            alt.Tooltip('Time', format='%d %b %Y, %H:%M'),
            alt.Tooltip('Drawdown', format='.6f')
        ])
        .add_selection(hover_selection)
    )
    drawdown_hover_rule = (
        alt.Chart(df)
        .mark_rule(opacity=0.25, size=0.25)
        .encode(x='Time')
        .transform_filter(hover_selection)
    )
    drawdown_hover_points = (
        drawdown_line
        .mark_point(filled=True, color='red')
        .encode(opacity=alt.condition(hover_selection, alt.value(1), alt.value(0)))
    )

    return alt.layer(
        drawdown_line, drawdown_hover_selectors, drawdown_hover_rule, drawdown_hover_points
    ).properties(
        title='Drawdown', width=1000, height=100
    )


def get_candles_chart(df: pd.DataFrame) -> alt.LayerChart:
    open_close_color = alt.condition(
        'datum.open <= datum.close', alt.value('#06982d'), alt.value('#ae1325')
    )
    base = alt.Chart(df).encode(alt.X('open_time:T'), color=open_close_color)
    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )
    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )
    return (rule + bar).properties(width=1200, height=700).interactive()
