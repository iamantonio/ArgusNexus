/**
 * ARGUSNEXUS V4 | Chart Manager
 * Plotly.js wrapper for Bloomberg-style candlestick charts
 */

class ChartManager {
    constructor() {
        this.currentSymbol = 'BTC-USD';
        this.currentTimeframe = 'FOUR_HOUR';
        this.chartData = null;
        this.position = null;
        this.isLightMode = false;

        // Color schemes
        this.darkColors = {
            background: '#0a0a0f',
            paper: '#101018',
            grid: '#1c1c28',
            text: '#5a5a68',
            textLight: '#a8a8b0',
            green: '#4af6c3',
            red: '#ff433d',
            orange: '#fb8b1e',
            blue: '#0068ff',
            white: '#ffffff'
        };

        this.lightColors = {
            background: '#f8f9fa',
            paper: '#ffffff',
            grid: '#e5e7eb',
            text: '#9ca3af',
            textLight: '#4b5563',
            green: '#059669',
            red: '#dc2626',
            orange: '#d97706',
            blue: '#2563eb',
            white: '#111827'
        };

        // Start with dark colors
        this.colors = this.darkColors;

        // Plotly layout config
        this.layout = {
            paper_bgcolor: this.colors.paper,
            plot_bgcolor: this.colors.background,
            font: {
                family: 'IBM Plex Mono, monospace',
                color: this.colors.text,
                size: 10
            },
            margin: { l: 50, r: 80, t: 20, b: 40 },
            xaxis: {
                gridcolor: this.colors.grid,
                showgrid: true,
                gridwidth: 1,
                linecolor: this.colors.grid,
                tickfont: { size: 9 },
                rangeslider: { visible: false }
            },
            yaxis: {
                gridcolor: this.colors.grid,
                showgrid: true,
                gridwidth: 1,
                linecolor: this.colors.grid,
                side: 'right',
                tickfont: { size: 9 },
                tickformat: ',.0f'
            },
            showlegend: false,
            dragmode: 'pan',
            hovermode: 'x unified'
        };

        // Plotly config
        this.config = {
            responsive: true,
            displayModeBar: false,
            scrollZoom: true
        };
    }

    /**
     * Render candlestick chart
     */
    async render(containerId, symbol, timeframe, position = null) {
        this.currentSymbol = symbol;
        this.currentTimeframe = timeframe;
        this.position = position;

        try {
            // Fetch candle data
            const data = await window.api.getCandles(symbol, {
                granularity: timeframe,
                limit: 150
            });

            if (!data.candles || data.candles.length === 0) {
                this.renderEmpty(containerId);
                return;
            }

            this.chartData = data.candles;

            // Build traces
            const traces = this.buildTraces(data.candles, position);

            // Render
            Plotly.newPlot(containerId, traces, this.layout, this.config);

            // Update current price overlay
            const latestPrice = data.candles[data.candles.length - 1].close;
            this.updatePriceOverlay(latestPrice);

        } catch (error) {
            console.error('[CHART] Render failed:', error);
            this.renderEmpty(containerId);
        }
    }

    /**
     * Build chart traces
     */
    buildTraces(candles, position) {
        const traces = [];

        // Convert timestamps to dates
        const timestamps = candles.map(c => new Date(c.timestamp * 1000));
        const opens = candles.map(c => c.open);
        const highs = candles.map(c => c.high);
        const lows = candles.map(c => c.low);
        const closes = candles.map(c => c.close);

        // Candlestick trace
        traces.push({
            type: 'candlestick',
            x: timestamps,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            increasing: {
                line: { color: this.colors.green, width: 1 },
                fillcolor: this.colors.green
            },
            decreasing: {
                line: { color: this.colors.red, width: 1 },
                fillcolor: this.colors.red
            },
            hoverinfo: 'x+text',
            text: candles.map(c =>
                `O: ${c.open.toLocaleString()}<br>` +
                `H: ${c.high.toLocaleString()}<br>` +
                `L: ${c.low.toLocaleString()}<br>` +
                `C: ${c.close.toLocaleString()}`
            )
        });

        // Position lines
        if (position) {
            const xRange = [timestamps[0], timestamps[timestamps.length - 1]];

            // Entry line (green dashed)
            if (position.entry_price) {
                traces.push(this.createHorizontalLine(
                    xRange,
                    position.entry_price,
                    this.colors.green,
                    'dash',
                    `ENTRY $${position.entry_price.toLocaleString()}`
                ));
            }

            // Stop line (red solid)
            if (position.stop_loss) {
                traces.push(this.createHorizontalLine(
                    xRange,
                    position.stop_loss,
                    this.colors.red,
                    'solid',
                    `STOP $${position.stop_loss.toLocaleString()}`
                ));
            }

            // Target line (blue dotted)
            if (position.take_profit) {
                traces.push(this.createHorizontalLine(
                    xRange,
                    position.take_profit,
                    this.colors.blue,
                    'dot',
                    `TARGET $${position.take_profit.toLocaleString()}`
                ));
            }
        }

        return traces;
    }

    /**
     * Create horizontal line trace
     */
    createHorizontalLine(xRange, yValue, color, dash, label) {
        return {
            type: 'scatter',
            mode: 'lines',
            x: xRange,
            y: [yValue, yValue],
            line: {
                color: color,
                width: dash === 'solid' ? 2 : 1,
                dash: dash
            },
            hoverinfo: 'text',
            text: label,
            showlegend: false
        };
    }

    /**
     * Update price overlay
     */
    updatePriceOverlay(price) {
        const overlay = document.getElementById('current-price');
        if (overlay) {
            overlay.textContent = `$${price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
        }
    }

    /**
     * Render empty state
     */
    renderEmpty(containerId) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div style="
                    height: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #5a5a68;
                    font-size: 11px;
                    letter-spacing: 1px;
                ">
                    AWAITING DATA FEED...
                </div>
            `;
        }
    }

    /**
     * Update chart with new data
     */
    async update(position = null) {
        await this.render(
            'price-chart',
            this.currentSymbol,
            this.currentTimeframe,
            position
        );
    }

    /**
     * Change symbol
     */
    async setSymbol(symbol) {
        this.currentSymbol = symbol;
        await this.update(this.position);
    }

    /**
     * Change timeframe
     */
    async setTimeframe(timeframe) {
        this.currentTimeframe = timeframe;
        await this.update(this.position);
    }

    /**
     * Set light/dark mode
     */
    setLightMode(isLight) {
        this.isLightMode = isLight;
        this.colors = isLight ? this.lightColors : this.darkColors;

        // Update layout with new colors
        this.layout.paper_bgcolor = this.colors.paper;
        this.layout.plot_bgcolor = this.colors.background;
        this.layout.font.color = this.colors.text;
        this.layout.xaxis.gridcolor = this.colors.grid;
        this.layout.xaxis.linecolor = this.colors.grid;
        this.layout.yaxis.gridcolor = this.colors.grid;
        this.layout.yaxis.linecolor = this.colors.grid;
    }
}

// Export singleton
window.chartManager = new ChartManager();
